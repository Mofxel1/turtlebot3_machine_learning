#!/usr/bin/env python3
#################################################################################
# SAC + LSTM Imitation Learning (Gösterimden Öğrenme)
# Orhan tarafindan Manuel Teleop Entegrasyonu ile yazildi.
# Güncelleme: Offline Pre-Training (Şok Eğitim) ve Yumuşak Kontrol eklendi.
#################################################################################

import collections
import datetime
import os
import random
import sys
import time
import numpy as np

# Klavye dinleme kütüphaneleri (Linux için)
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, LayerNormalization, LeakyReLU, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# GPU ayari
tf.config.set_visible_devices([], 'GPU')

# Terminal ayarlarini yedekle
settings = termios.tcgetattr(sys.stdin)

def get_key():
    """ Sistemi dondurmadan (non-blocking) klavyeden basilan tusu okur """
    tty.setraw(sys.stdin.fileno())
    # Döngü hızı ROS fizik motoruyla senkronize edildi (20 Hz)
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05) 
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class SACLSTMTeleopAgent(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('sac_lstm_teleop_agent')

        self.action_size = 2 # [Linear, Angular]
        self.stage = int(stage_num)
        
        # --- İNSAN KONTROLÜ AYARLARI ---
        self.autonomous_mode = False  # Baslangicta İnsan kontrolünde
        self.manual_lin = 0.0
        self.manual_ang = 0.0
        
        # --- Frame Stacking ---
        self.stack_size = 3         
        self.raw_state_size = 26    
        self.state_size = self.raw_state_size * self.stack_size 
        self.stacked_state = np.zeros((1, self.state_size), dtype=np.float32)
        self.is_stack_full = False

        self.max_training_episodes = int(max_training_episodes)

        # SAC Ayarlari
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        # Alpha (Merak) sönümleme hızını 10 kat yavaşlattık (Erken ezberlemeyi önlemek için)
        self.alpha_lr = 0.00003 
        self.batch_size = 64
        self.min_replay_memory_size = 1000 
        
        self.target_entropy = -np.prod((self.action_size,)).astype(np.float32)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha_optimizer = Adam(learning_rate=self.alpha_lr)

        self.replay_memory = collections.deque(maxlen=100000)

        # Aglari kur
        self.actor = self.build_actor()
        self.critic_1 = self.build_critic()
        self.critic_2 = self.build_critic()
        self.target_critic_1 = self.build_critic()
        self.target_critic_2 = self.build_critic()
        
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_1_optimizer = Adam(learning_rate=self.critic_lr)
        self.critic_2_optimizer = Adam(learning_rate=self.critic_lr)

        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        print("\n" + "="*50)
        print("🎮 TELEOP (MANUEL EĞİTİM) MODU BAŞLATILDI!")
        print("Kontroller: W(Gaz), S(Fren), X(Yavaşla), A(Sol), D(Sağ)")
        print("Otomatik Eğitime (Yapay Zeka) geçmek için 'o' tuşuna bas!")
        print("="*50 + "\n")

        self.process()

    # --- TENSORFLOW VE AG YAPILARI ---
    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        reshaped_state = Reshape((self.stack_size, self.raw_state_size))(state_input)
        x = LSTM(256, return_sequences=False)(reshaped_state)
        x = LayerNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dense(128)(x)
        x = LayerNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        mean = Dense(self.action_size, activation='linear')(x)
        log_std = Dense(self.action_size, activation='linear')(x)
        return Model(inputs=state_input, outputs=[mean, log_std])

    def build_critic(self):
        state_input = Input(shape=(self.state_size,))
        action_input = Input(shape=(self.action_size,))
        reshaped_state = Reshape((self.stack_size, self.raw_state_size))(state_input)
        s = LSTM(256, return_sequences=False)(reshaped_state)
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
        action = tf.tanh(sampled_action).numpy()[0]
        
        linear_val = np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0) 
        angular_val = np.clip(action[1], -1.0, 1.0)
        return [linear_val, angular_val]

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
        self.critic_1_optimizer.apply_gradients(zip(c1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(c2_grads, self.critic_2.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_actions, log_probs = self.sample_action_and_log_prob(states)
            q1_new = self.critic_1([states, new_actions])
            q2_new = self.critic_2([states, new_actions])
            q_min_new = tf.minimum(q1_new, q2_new)
            actor_loss = tf.reduce_mean(alpha * log_probs - q_min_new)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy))
        
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        
        self.update_target_networks()

    def update_target_networks(self):
        for target_weight, weight in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))
        for target_weight, weight in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def train_model(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        states = np.array([i[0][0] for i in batch], dtype=np.float32)
        actions_raw = np.array([i[1] for i in batch], dtype=np.float32)
        actions = np.zeros_like(actions_raw)
        actions[:, 0] = (actions_raw[:, 0] * 2.0) - 1.0 
        actions[:, 1] = actions_raw[:, 1]
        
        rewards = np.array([i[2] for i in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([i[3][0] for i in batch], dtype=np.float32)
        dones = np.array([float(i[4]) for i in batch], dtype=np.float32).reshape(-1, 1)

        self.train_step(tf.convert_to_tensor(states), tf.convert_to_tensor(actions), 
                        tf.convert_to_tensor(rewards), tf.convert_to_tensor(next_states), 
                        tf.convert_to_tensor(dones))

    # --- ROS VE ORTAM ISLEMLERI ---
    def get_stacked_state(self, new_state):
        new_state = np.asarray(new_state, dtype=np.float32).reshape(self.raw_state_size)
        if not self.is_stack_full:
            for i in range(self.stack_size):
                self.stacked_state[0, i*self.raw_state_size : (i+1)*self.raw_state_size] = new_state
            self.is_stack_full = True
        else:
            self.stacked_state[0, :-self.raw_state_size] = self.stacked_state[0, self.raw_state_size:]
            self.stacked_state[0, -self.raw_state_size:] = new_state
        return self.stacked_state

    def append_sample(self, transition):
        self.replay_memory.append(transition)

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
            return self.get_stacked_state(future.result().state)
        return np.zeros((1, self.state_size))

    def step(self, action):
        req = Dqn.Request()
        req.action = [float(action[0]), float(action[1])]
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return self.get_stacked_state(future.result().state), future.result().reward, future.result().done
        return np.zeros((1, self.state_size)), 0, True

    def process(self):
        self.env_make()
        time.sleep(1.0)

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            self.reset_environment_client.call_async(Dqn.Request())
            time.sleep(1.0) 
            
            state = self.reset_environment()
            local_step = 0
            score = 0

            while True:
                # --- KLAVYE DINLEYICISI ---
                key = get_key()
                
                if key == 'o' and not self.autonomous_mode:
                    self.autonomous_mode = True
                    print("\n" + "🧠"*15)
                    print("🧠 BEYİN ŞOK EĞİTİME (PRE-TRAINING) ALINIYOR! LÜTFEN BEKLEYİN...")
                    print("Senin sürüş kayıtların sindiriliyor, bu birkaç saniye sürebilir...")
                    
                    # 1000 döngülük hızlı eğitim (Senin sürüşünü klonluyor)
                    for _ in range(1000):
                        if len(self.replay_memory) > self.min_replay_memory_size:
                            self.train_model()
                            
                    print("\n" + "🚀"*15)
                    print("🚀 OTONOM MOD AKTİF! YAPAY ZEKA DİREKSİYONDA!")
                    print("🚀"*15 + "\n")
                elif key == '\x03': # Ctrl+C basilmissa
                    break

                if not self.autonomous_mode:
                    # İnsan Sürüşü (Daha yumuşak ivmelenme ve düşük max hız)
                    if key == 'w': self.manual_lin = min(self.manual_lin + 0.1, 1.0)
                    elif key == 'x': self.manual_lin = max(self.manual_lin - 0.1, 0.0)
                    elif key == 'a': self.manual_ang = min(self.manual_ang + 0.1, 1.0)
                    elif key == 'd': self.manual_ang = max(self.manual_ang - 0.1, -1.0)
                    elif key == 's': 
                        self.manual_lin = 0.0
                        self.manual_ang = 0.0
                    
                    action = [self.manual_lin, self.manual_ang]
                    sys.stdout.write(f"\r[MANUEL] İleri: {action[0]:.2f}, Dönüş: {action[1]:.2f}   ")
                    sys.stdout.flush()
                else:
                    # Yapay Zeka Sürüşü
                    action = self.get_action(state)

                # Aksiyonu ortama gonder
                next_state, reward, done = self.step(action)
                score += reward

                # Aniyi hafizaya yaz 
                self.append_sample((state, action, reward, next_state, done))

                # Egitim (Sen manuel sürerken bile arka planda ağları günceller)
                if len(self.replay_memory) > self.min_replay_memory_size:
                    self.train_model()

                state = next_state
                local_step += 1

                if done:
                    current_alpha = tf.exp(self.log_alpha).numpy()
                    mod_str = "OTONOM" if self.autonomous_mode else "MANUEL"
                    print(f"\nBölüm: {episode} | Mod: {mod_str} | Skor: {score:.2f} | Adım: {local_step} | Hafıza: {len(self.replay_memory)}")
                    break

def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '3'
    max_training_episodes = args[2] if len(args) > 2 else '5000'
    
    rclpy.init(args=args)
    agent = SACLSTMTeleopAgent(stage_num, max_training_episodes)
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()