#!/usr/bin/env python3
#################################################################################
# SAC + LSTM Otonom Ralli Pilotu (Stage 3 İçin Tam Otonom Versiyon)
# Orhan tarafindan modifiye edilmis ve optimize edilmistir.
#################################################################################

import collections
import datetime
import os
import random
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, LayerNormalization, LeakyReLU, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from turtlebot3_msgs.srv import Dqn

# Olası çakışmaları önlemek için CPU kullanımı (GPU varsa burayı silebilirsin)
tf.config.set_visible_devices([], 'GPU')

class SACLSTMAgentNode(Node):
    def __init__(self, stage_num, max_training_episodes):
        super().__init__('sac_lstm_agent')

        self.action_size = 2 
        self.stage = int(stage_num)
        self.train_mode = True  
        
        # --- AŞAMA 2: ALGI VE HAFIZA GÜNCELLEMESİ ---
        self.stack_size = 5         # Hafıza penceresini 5 kareye çıkardık (Hareket tahmini için)
        self.raw_state_size = 74    # 72 Lidar Işını + 2 Hedef (Mesafe, Açı)
        self.state_size = self.raw_state_size * self.stack_size # Toplam 370 boyut
        
        self.stacked_state = np.zeros((1, self.state_size), dtype=np.float32)
        self.is_stack_full = False

        self.max_training_episodes = int(max_training_episodes)

        # --- AŞAMA 3: EĞİTİM HİPERPARAMETRELERİ ---
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.min_replay_memory_size = 1000 
        
        self.target_entropy = -float(self.action_size) # Entropi hedefi otomatik ayarlandı (-2.0)
        # DUZELTME: log_alpha baslangic degeri 0.0 yerine 0.5 (alpha=1.65), kesfin hizli
        # azalmasi onleniyor. Alpha_opt learning rate de dusuruldu.
        self.log_alpha = tf.Variable(0.5, dtype=tf.float32)
        
        # Hafıza (Replay Buffer)
        self.step_counter = 0
        self.replay_memory = collections.deque(maxlen=100000)

        # Ağların Kurulumu
        self.actor = self.build_actor()
        self.critic_1 = self.build_critic()
        self.critic_2 = self.build_critic()
        self.target_critic_1 = self.build_critic()
        self.target_critic_2 = self.build_critic()
        
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_opt = Adam(learning_rate=0.0003)
        self.critic_1_opt = Adam(learning_rate=0.0003)
        self.critic_2_opt = Adam(learning_rate=0.0003)
        # DUZELTME: alpha_opt lr 0.00003 -> 0.000005, entropi cok hizli dusuyordu
        self.alpha_opt = Adam(learning_rate=0.00001)

        # Loglama ve Kayıt Dizini
        self.load_episode = 0
        home_dir = os.path.expanduser('~')
        self.model_dir_path = os.path.join(home_dir, 'turtlebot3_ws', 'src', 'turtlebot3_machine_learning', 'turtlebot3_dqn', 'saved_model')
        
        self.log_dir = os.path.join(self.model_dir_path, 'logs_sac', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        # ROS 2 Servis İstemcileri
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.get_logger().info(f"Otonom SAC+LSTM Baslatiliyor... Mod: Stage {self.stage} | Giris: {self.state_size}")
        self.process()

    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        # State'i zaman serisine (5 adım, 74 özellik) çevirip LSTM'e veriyoruz
        reshaped_state = Reshape((self.stack_size, self.raw_state_size))(state_input)
        
        x = LSTM(128, return_sequences=False)(reshaped_state)
        x = LayerNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        
        x = Dense(128)(x)
        x = LayerNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        
        last_init = RandomUniform(minval=-0.003, maxval=0.003)
        mean = Dense(self.action_size, activation='linear', kernel_initializer=last_init)(x)
        log_std = Dense(self.action_size, activation='linear', kernel_initializer=last_init)(x)
        return Model(inputs=state_input, outputs=[mean, log_std])

    def build_critic(self):
        state_input = Input(shape=(self.state_size,))
        action_input = Input(shape=(self.action_size,))
        
        # State'i zaman serisine çevirip LSTM'e veriyoruz
        reshaped_state = Reshape((self.stack_size, self.raw_state_size))(state_input)
        s = LSTM(128, return_sequences=False)(reshaped_state)
        s = LayerNormalization()(s)
        s = LeakyReLU(negative_slope=0.1)(s)
        
        x = Concatenate()([s, action_input])
        x = Dense(256)(x)
        x = LayerNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        
        x = Dense(128)(x)
        x = LayerNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        
        q_value = Dense(1, activation='linear')(x)
        return Model(inputs=[state_input, action_input], outputs=q_value)

    def get_action(self, state):
        mean, log_std = self.actor(state)
        log_std = tf.clip_by_value(log_std, -20.0, 2.0)
        std = tf.exp(log_std)
        
        noise = tf.random.normal(shape=mean.shape)
        sampled_action = mean + std * noise
        
        # --- AŞAMA 1 UYUMU: Ajan daima [-1, 1] arası eylem üretir ---
        action = tf.tanh(sampled_action).numpy()[0]
        return [float(action[0]), float(action[1])]

    def sample_action_and_log_prob(self, state):
        mean, log_std = self.actor(state)
        log_std = tf.clip_by_value(log_std, -20.0, 2.0)
        std = tf.exp(log_std)
        noise = tf.random.normal(shape=mean.shape)
        sampled_action = mean + std * noise
        action = tf.tanh(sampled_action)
        
        log_prob = -0.5 * (((sampled_action - mean) / (std + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        log_prob -= tf.reduce_sum(tf.math.log(1.0 - action**2 + 1e-6), axis=1, keepdims=True)
        return action, log_prob

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        alpha = tf.exp(self.log_alpha)
        
        next_actions, next_log_probs = self.sample_action_and_log_prob(next_states)
        target_q1 = self.target_critic_1([next_states, next_actions])
        target_q2 = self.target_critic_2([next_states, next_actions])
        
        target_q_min = tf.minimum(target_q1, target_q2) - alpha * next_log_probs
        target_q = rewards + (1.0 - dones) * self.gamma * target_q_min
        
        with tf.GradientTape(persistent=True) as tape:
            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])
            critic_1_loss = tf.reduce_mean(tf.square(target_q - current_q1))
            critic_2_loss = tf.reduce_mean(tf.square(target_q - current_q2))

        c1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        c2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_opt.apply_gradients(zip(c1_grads, self.critic_1.trainable_variables))
        self.critic_2_opt.apply_gradients(zip(c2_grads, self.critic_2.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_actions, log_probs = self.sample_action_and_log_prob(states)
            q1_new = self.critic_1([states, new_actions])
            q2_new = self.critic_2([states, new_actions])
            q_min_new = tf.minimum(q1_new, q2_new)
            actor_loss = tf.reduce_mean(alpha * log_probs - q_min_new)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy))
        
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_opt.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        
        self.update_target_networks()
        
        return critic_1_loss, critic_2_loss, actor_loss

    def update_target_networks(self):
        for target_weight, weight in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))
        for target_weight, weight in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def train_model(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        states = np.array([i[0][0] for i in batch], dtype=np.float32)
        actions = np.array([i[1] for i in batch], dtype=np.float32) 
        rewards = np.array([i[2] for i in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([i[3][0] for i in batch], dtype=np.float32)
        dones = np.array([float(i[4]) for i in batch], dtype=np.float32).reshape(-1, 1)

        c1_loss, c2_loss, a_loss = self.train_step(tf.convert_to_tensor(states), 
                                                   tf.convert_to_tensor(actions), 
                                                   tf.convert_to_tensor(rewards), 
                                                   tf.convert_to_tensor(next_states), 
                                                   tf.convert_to_tensor(dones))
        
        self.current_critic_loss = (c1_loss + c2_loss) / 2.0
        self.current_actor_loss = a_loss

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            pass
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            pass
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        
        self.is_stack_full = False 
        
        if future.result() is not None:
            raw_state = future.result().state 
            new_state = np.asarray(raw_state, dtype=np.float32).reshape(self.raw_state_size)
            for i in range(self.stack_size):
                start_idx = i * self.raw_state_size
                end_idx = start_idx + self.raw_state_size
                self.stacked_state[0, start_idx:end_idx] = new_state
            self.is_stack_full = True
            state = self.stacked_state.copy()
        else:
            state = np.zeros((1, self.state_size))
        return state

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            raw_next_state = future.result().state
            new_state = np.asarray(raw_next_state, dtype=np.float32).reshape(self.raw_state_size)
            
            # Kaydırma işlemi (Frame Stacking)
            self.stacked_state[0, :-self.raw_state_size] = self.stacked_state[0, self.raw_state_size:]
            self.stacked_state[0, -self.raw_state_size:] = new_state
            
            next_state = self.stacked_state.copy()
            reward = future.result().reward
            done = future.result().done
        else:
            next_state = np.zeros((1, self.state_size))
            reward = 0
            done = True
        return next_state, reward, done

    def process(self):
        self.env_make()
        time.sleep(1.0)

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            # DUZELTME: cift reset kaldirildi. Asagidaki reset_environment() zaten
            # reset_environment_client cagrisi yapiyor. Iki kez cagirmak
            # state senkronunu bozuyor ve 3-adimlik anlik carpismaya yol aciyordu.
            state = self.reset_environment()
            local_step = 0
            score = 0

            while True:
                time.sleep(0.05)
                
                local_step += 1
                self.step_counter += 1

                action = self.get_action(state)
                next_state, reward, done = self.step(action)
                score += reward

                # Hafızaya ekleme
                self.replay_memory.append((state, action, reward, next_state, done))

                # Eğitim adımı
                if self.train_mode and len(self.replay_memory) > self.min_replay_memory_size:
                    self.train_model()

                state = next_state

                if done:
                    current_alpha = np.exp(self.log_alpha.numpy())
                    basari_mesaji = "HEDEFE ULASILDI!" if reward >= 50.0 else "Hata yapildi veya Sure Bitti."
                    
                    print(f"Bölüm: {episode} | Skor: {score:.2f} | Adım: {local_step} | Aktif Entropi: {current_alpha:.4f} -> {basari_mesaji}")
                    
                    with self.summary_writer.as_default():
                        tf.summary.scalar('Episode Score', score, step=episode)
                        tf.summary.scalar('Aktif Alpha (Entropi)', current_alpha, step=episode)
                        
                        if hasattr(self, 'current_actor_loss'):
                            tf.summary.scalar('Actor Loss', self.current_actor_loss, step=episode)
                            tf.summary.scalar('Critic Loss', self.current_critic_loss, step=episode)
                            
                    self.summary_writer.flush()
                    
                    if episode % 10 == 0:
                        if not os.path.exists(self.model_dir_path):
                            os.makedirs(self.model_dir_path)
                        actor_path = os.path.join(self.model_dir_path, f'stage{self.stage}_sac_actor.keras')
                        self.actor.save(actor_path)
                    break

def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '3'
    max_training_episodes = args[2] if len(args) > 2 else '5000'
    
    rclpy.init(args=args)
    sac_agent = SACLSTMAgentNode(stage_num, max_training_episodes)
    
    try:
        rclpy.spin(sac_agent)
    except KeyboardInterrupt:
        pass
    finally:
        sac_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
