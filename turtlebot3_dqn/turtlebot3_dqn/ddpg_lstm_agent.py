#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
# Modified by Orhan for Continuous Control with Frame Stacking (3 Frames)
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
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from turtlebot3_msgs.srv import Dqn

# GPU ayarı
tf.config.set_visible_devices([], 'GPU')

class OUNoise:
    """Ornstein-Uhlenbeck süreci ile zamansal ilişkili gürültü üretici."""
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state

class DDPGAgent(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('ddpg_agent')

        self.action_size = 2 # [Linear Velocity, Angular Velocity]
        
        # --- GÜRÜLTÜ AYARLARI ---
        self.ou_noise = OUNoise(action_dimension=self.action_size, mu=0.0, theta=0.15, sigma=0.2)

        self.stage = int(stage_num)
        self.train_mode = True  
        
        # --- GÜNCELLEME 1: Frame Stacking Ayarları ---
        self.stack_size = 3         # Hafızada tutulacak kare sayısı (3 an)
        self.raw_state_size = 26    # Robotun gönderdiği tekil veri (Lidar + Info)
        self.state_size = self.raw_state_size * self.stack_size # 26 * 3 = 78 (Ağ Girişi)
        
        # Kayan Pencere (Hafıza Kuyruğu) - Otomatik olarak en eskiyi siler
        self.stacked_state = np.zeros((1, self.state_size), dtype=np.float32)
        self.is_stack_full = False

        self.max_training_episodes = int(max_training_episodes)

        # Hiperparametreler
        self.discount_factor = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.001 
        self.batch_size = 64
        self.min_replay_memory_size = 1000 

        self.step_counter = 0
        self.replay_memory = collections.deque(maxlen=100000)

        # Gürültü Ayarları
        self.epsilon = 1.0          
        self.epsilon_decay_param = 0.005 
        self.epsilon_min = 0.05     

        # Modeller
        self.actor_model = self.create_actor_model()
        self.critic_model = self.create_critic_model()
        self.target_actor = self.create_actor_model()
        self.target_critic = self.create_critic_model()
        
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

        # Kayıt Yolları
        self.load_model_flag = False 
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        # --- TENSORBOARD LOG AYARLARI ---
        self.log_dir = os.path.join(self.model_dir_path, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.actor_path = os.path.join(self.model_dir_path, f'stage{self.stage}_actor.keras')
        self.critic_path = os.path.join(self.model_dir_path, f'stage{self.stage}_critic.keras')

        if self.load_model_flag:
            if os.path.exists(self.actor_path):
                # Model boyutları değiştiği için eski modeller hata verebilir!
                try:
                    self.actor_model = load_model(self.actor_path)
                    self.critic_model = load_model(self.critic_path)
                    self.target_actor.set_weights(self.actor_model.get_weights())
                    self.target_critic.set_weights(self.critic_model.get_weights())
                    print("Kayıtlı modeller yüklendi!")
                except:
                    print("Eski modellerin boyutu uymadı, sıfırdan başlanıyor...")
            else:
                print("Model dosyası bulunamadı, sıfırdan başlanıyor.")

        # ROS Servisleri
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.process()

    # --- GÜNCELLEME 2: Stack Oluşturma Fonksiyonu ---
    def get_stacked_state(self, new_state):
    # Gelen veriyi (1, 26) yerine doğrudan düz bir vektöre (26,) ve float32'ye çevir
        new_state = np.asarray(new_state, dtype=np.float32).reshape(self.raw_state_size)
        
        if not self.is_stack_full:
            # Bölüm (Episode) başındayız. Hafızanın içi boş.
            # İlk gelen kareyi (frame), ağın girişindeki 3 yere de kopyalıyoruz.
            for i in range(self.stack_size):
                start_idx = i * self.raw_state_size
                end_idx = start_idx + self.raw_state_size
                self.stacked_state[0, start_idx:end_idx] = new_state
            self.is_stack_full = True
        else:
            # Hafıza zaten dolu. Yeni veri geldi.
            # 1. Eski verileri Sola Kaydır: [t-2, t-1, t] -> [t-1, t, t]
            self.stacked_state[0, :-self.raw_state_size] = self.stacked_state[0, self.raw_state_size:]
            
            # 2. En sağdaki boşluğa yeni veriyi (t+1) yaz: [t-1, t, t+1]
            self.stacked_state[0, -self.raw_state_size:] = new_state
            
        return self.stacked_state
    # ------------------------------------------------

    def process(self):
        self.env_make()
        time.sleep(1.0)

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            
            if self.train_mode:
                decay_factor = 1.0 + (episode * self.epsilon_decay_param)
                self.epsilon = 1.0 / decay_factor
                self.epsilon = max(self.epsilon, self.epsilon_min)


            self.reset_environment_client.call_async(Dqn.Request())
            time.sleep(1.0) 
            
            # Reset sonrası ilk stackli durumu al
            state = self.reset_environment()
            
            local_step = 0
            score = 0

            while True:
                local_step += 1
                self.step_counter += 1

                action = self.get_action(state)
                next_state, reward, done = self.step(action)
                score += reward

                self.append_sample((state, action, reward, next_state, done))

                if self.train_mode and len(self.replay_memory) > self.min_replay_memory_size:
                    self.train_model()

                state = next_state

                if done:
                    print(f"Episode: {episode}, Score: {score:.2f}, Steps: {local_step}, Epsilon: {self.epsilon:.3f}, Memory: {len(self.replay_memory)}")
                    
                    # --- TENSORBOARD'A YAZDIR ---
                    with self.summary_writer.as_default():
                        tf.summary.scalar('Episode Score (Reward)', score, step=episode)
                        tf.summary.scalar('Epsilon (Exploration)', self.epsilon, step=episode)
                    self.summary_writer.flush()
                    # ----------------------------


                    if episode % 10 == 0:
                        if not os.path.exists(self.model_dir_path):
                            os.makedirs(self.model_dir_path)
                        self.actor_model.save(self.actor_path)
                        self.critic_model.save(self.critic_path)
                        print("Modeller kaydedildi (.keras)")
                    break

    def create_actor_model(self):
        state_input = Input(shape=(self.state_size,))
        
        # 1. 78 boyutlu düz veriyi, (3 zaman adımı, 26 Lidar verisi) formatına çeviriyoruz
        reshaped_state = Reshape((self.stack_size, self.raw_state_size))(state_input)
        
        # 2. LSTM Katmanı: Zamanı ve hareketin yönünü kavrar
        h1 = LSTM(256)(reshaped_state)
        h1 = LayerNormalization()(h1)
        h1 = LeakyReLU(negative_slope=0.1)(h1)
        
        # 3. Karar Katmanı
        h2 = Dense(128)(h1)
        h2 = LayerNormalization()(h2)
        h2 = LeakyReLU(negative_slope=0.1)(h2)
        
        last_init = RandomUniform(minval=-0.003, maxval=0.003)
        linear_vel = Dense(1, activation='sigmoid', kernel_initializer=last_init, name='linear')(h2) 
        angular_vel = Dense(1, activation='tanh', kernel_initializer=last_init, name='angular')(h2)  
        
        output = Concatenate()([linear_vel, angular_vel])
        return Model(state_input, output)

    def create_critic_model(self):
        state_input = Input(shape=(self.state_size,))
        action_input = Input(shape=(self.action_size,))
        
        # 1. Çevrenin zaman içindeki değişimini anla
        reshaped_state = Reshape((self.stack_size, self.raw_state_size))(state_input)
        s1 = LSTM(256)(reshaped_state)
        s1 = LayerNormalization()(s1)
        s1 = LeakyReLU(negative_slope=0.1)(s1)
        
        # 2. Şoförün o anki hamlesiyle LSTM'in çıkardığı zaman algısını birleştir
        concat = Concatenate()([s1, action_input])
        
        h1 = Dense(256)(concat)
        h1 = LayerNormalization()(h1)
        h1 = LeakyReLU(negative_slope=0.1)(h1)
        
        h2 = Dense(128)(h1)
        h2 = LayerNormalization()(h2)
        h2 = LeakyReLU(negative_slope=0.1)(h2)
        
        output = Dense(1, activation='linear')(h2) 
        
        return Model([state_input, action_input], output)

    def get_action(self, state):
        preds = self.actor_model.predict(state, verbose=0)[0]
        linear_val = float(preds[0])
        angular_val = float(preds[1])
        
        if self.train_mode:
            # OU Noise'dan 2 boyutlu [linear, angular] ilişkili gürültüyü al
            noise = self.ou_noise.noise() * self.epsilon
            
            # Gürültüyü aksiyonlara ekle
            linear_val += noise[0]
            angular_val += noise[1]
            
        # Fiziksel limitleri koru
        return [np.clip(linear_val, 0.0, 1.0), np.clip(angular_val, -1.0, 1.0)]

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        # Huber Loss objesini oluştur (Büyük hataları dizginlemek için delta=1.0 idealdir)
        huber_loss = tf.keras.losses.Huber(delta=1.0)

        # --- 1. CRITIC GÜNCELLEMESİ ---
        with tf.GradientTape() as tape:
            # DİKKAT: Hedef ağlar eğitim yapmaz! training=False olmalı.
            target_actions = self.target_actor(next_states, training=False)
            target_q_values = self.target_critic([next_states, target_actions], training=False)
            
            # Bellman Denklemi: y = r + gamma * Q_target * (1 - done)
            y = rewards + (1 - dones) * self.discount_factor * target_q_values
            
            # Ana Critic ağı eğitimi yaptığı için training=True
            critic_value = self.critic_model([states, actions], training=True)
            
            # MSE yerine Huber Loss ile gradyan patlamalarını engelliyoruz
            critic_loss = huber_loss(y, critic_value)

        # Critic ağırlıklarını güncelle
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # --- 2. ACTOR GÜNCELLEMESİ ---
        with tf.GradientTape() as tape:
            # Actor yeni eylemler planlıyor (training=True)
            new_actions = self.actor_model(states, training=True)
            
            # Critic bu yeni eylemlere puan veriyor (training=False çünkü critic'i güncellemıyoruz, sadece notuna bakıyoruz)
            critic_value = self.critic_model([states, new_actions], training=False)
            
            # Actor, Critic'in verdiği puanı maksimize etmeye çalışır (Kayıp negatif olmalı)
            actor_loss = -tf.math.reduce_mean(critic_value)

        # Actor ağırlıklarını güncelle
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
    
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def train_model(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        
        states = np.array([i[0][0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch]).reshape(-1, 1)
        next_states = np.array([i[3][0] for i in batch])
        dones = np.array([float(i[4]) for i in batch]).reshape(-1, 1)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        self.train_step(states, actions, rewards, next_states, dones)
        self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Environment make client failed, retrying...')
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset environment client failed, retrying...')
        
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        
        # --- GÜNCELLEME 3: Stack'i Sıfırla ve Doldur ---
        self.is_stack_full = False # Bölüm başı hafızayı sil
        self.ou_noise.reset() # OU Gürültüsünü sıfırla
        
        if future.result() is not None:
            raw_state = future.result().state 
            state = self.get_stacked_state(raw_state) # 3'lü paket yap
        else:
            self.get_logger().error('Service call failed')
            state = np.zeros((1, self.state_size))

        return state

    def step(self, action):
        req = Dqn.Request()
        req.action = [float(action[0]), float(action[1])]

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for service...')

        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        # --- GÜNCELLEME 4: Yeni durumu Stack'e ekle ---
        if future.result() is not None:
            raw_next_state = future.result().state
            next_state = self.get_stacked_state(raw_next_state) # 3'lü paket
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error('Service call failed')
            next_state = np.zeros((1, self.state_size))
            reward = 0
            done = True

        return next_state, reward, done

def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '5000'
    
    rclpy.init(args=args)
    ddpg_agent = DDPGAgent(stage_num, max_training_episodes)
    rclpy.spin(ddpg_agent)

    ddpg_agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
