#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
# Modified by Orhan for Continuous Control (DDPG Architecture)
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
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from turtlebot3_msgs.srv import Dqn

# GPU ayarı (İsteğe bağlı, CPU kullanıyorsan burası kalabilir)
tf.config.set_visible_devices([], 'GPU')

class DDPGAgent(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('ddpg_agent')

        self.stage = int(stage_num)
        self.train_mode = False  # EĞİTİM MODU AÇIK!
        
        self.state_size = 14
        self.action_size = 2 # [Linear Velocity, Angular Velocity]
        self.max_training_episodes = int(max_training_episodes)

        # Hiperparametreler (DDPG Ayarları)
        self.discount_factor = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.001 # Hedef ağları yavaş güncelleme oranı (Soft Update)
        self.batch_size = 64
        self.min_replay_memory_size = 1000 # Hafıza bu kadar dolmadan eğitime başlama

        self.step_counter = 0
        self.replay_memory = collections.deque(maxlen=100000)

        # --- GÜRÜLTÜ VE EPSILON AYARLARI (Time-Based Decay) ---
        self.epsilon = 1.0          # Başlangıç gürültüsü (%100)
        self.epsilon_decay_param = 0.005 # Azalma hızı (Ne kadar büyükse o kadar hızlı düşer)
        self.epsilon_min = 0.05     # Minimum gürültü (%5 merak payı)
        # ------------------------------------------------------

        # --- MODELLERİ OLUŞTUR (ACTOR & CRITIC) ---
        
        # 1. Ana Modeller
        self.actor_model = self.create_actor_model()
        self.critic_model = self.create_critic_model()
        
        # 2. Hedef (Target) Modeller (Eğitim istikrarı için)
        self.target_actor = self.create_actor_model()
        self.target_critic = self.create_critic_model()
        
        # Ağırlıkları eşitle
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Optimizasyoncular (Öğretmenler)
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

        # Model Yükleme/Kaydetme Yolları
        self.load_model_flag = True 
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        
        # Dosya uzantısını .keras yaptık (Warning kalksın diye)
        self.actor_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_actor.keras'
        )
        self.critic_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_critic.keras'
        )

        if self.load_model_flag:
            if os.path.exists(self.actor_path):
                self.actor_model = load_model(self.actor_path)
                self.critic_model = load_model(self.critic_path)
                # Targetları da güncelle
                self.target_actor.set_weights(self.actor_model.get_weights())
                self.target_critic.set_weights(self.critic_model.get_weights())
                print("Kayıtlı modeller yüklendi!")
            else:
                print("Model dosyası bulunamadı, sıfırdan başlanıyor.")

        # ROS Servisleri
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            
            # --- ZAMAN BAZLI EPSILON HESAPLAMA ---
            # Formül: 1 / (1 + (Bölüm * Katsayı))
            if self.train_mode:
                decay_factor = 1.0 + (episode * self.epsilon_decay_param)
                self.epsilon = 1.0 / decay_factor
                # Minimumun altına inmesin
                self.epsilon = max(self.epsilon, self.epsilon_min)
            # -------------------------------------

            # 1. Resetle ve Bekle (Hayalet veri olmasın diye)
            self.reset_environment_client.call_async(Dqn.Request())
            time.sleep(1.0) 
            state = self.reset_environment()
            
            local_step = 0
            score = 0

            while True:
                local_step += 1
                self.step_counter += 1

                # 1. Eylem Seç (Gürültülü veya Gürültüsüz)
                action = self.get_action(state)

                # 2. Uygula
                next_state, reward, done = self.step(action)
                score += reward

                # 3. Hafızaya Kaydet
                self.append_sample((state, action, reward, next_state, done))

                # 4. EĞİTİM (Öğrenme burada gerçekleşiyor)
                if self.train_mode and len(self.replay_memory) > self.min_replay_memory_size:
                    self.train_model()

                state = next_state

                if done:
                    # Log çıktısına Epsilon'u da ekledik
                    print(f"Episode: {episode}, Score: {score:.2f}, Steps: {local_step}, Epsilon: {self.epsilon:.3f}, Memory: {len(self.replay_memory)}")
                    
                    # Her 10 bölümde bir kaydet (Yeni formatta)
                    if episode % 10 == 0:
                        if not os.path.exists(self.model_dir_path):
                            os.makedirs(self.model_dir_path)
                        self.actor_model.save(self.actor_path)
                        self.critic_model.save(self.critic_path)
                        print("Modeller kaydedildi (.keras)")
                    break

                # Bilgisayarı kitlememek için minik bekleme
                # time.sleep(0.01) 

    def create_actor_model(self):
        """Actor: Durum -> Eylem (Gaz, Direksiyon)"""
        state_input = Input(shape=(self.state_size,))
        h1 = Dense(512, activation='relu')(state_input)
        h2 = Dense(512, activation='relu')(h1)
        h3 = Dense(256, activation='relu')(h2)
        
        linear_vel = Dense(1, activation='sigmoid', name='linear')(h3) # 0..1
        angular_vel = Dense(1, activation='tanh', name='angular')(h3)  # -1..1
        
        output = Concatenate()([linear_vel, angular_vel])
        return Model(state_input, output)

    def create_critic_model(self):
        """Critic: Durum + Eylem -> Puan (Q-Value)"""
        state_input = Input(shape=(self.state_size,))
        action_input = Input(shape=(self.action_size,))
        
        # State ve Action'ı birleştir
        concat = Concatenate()([state_input, action_input])
        
        h1 = Dense(512, activation='relu')(concat)
        h2 = Dense(512, activation='relu')(h1)
        h3 = Dense(256, activation='relu')(h2)
        
        output = Dense(1, activation='linear')(h3) # Q Değeri (Örn: 15.4 veya -50)
        return Model([state_input, action_input], output)

    def get_action(self, state):
        # Tahmin
        preds = self.actor_model.predict(state, verbose=0)[0]
        linear_val = float(preds[0])
        angular_val = float(preds[1])
        
        # Gürültü (Noise) - Epsilon ile ölçeklenmiş
        if self.train_mode:
            # Gürültü şiddetini epsilon ile çarpıyoruz
            # Epsilon düştükçe gürültü azalır
            noise_linear = np.random.normal(0, 0.1) * self.epsilon
            noise_angular = np.random.normal(0, 0.2) * self.epsilon
            
            linear_val += noise_linear
            angular_val += noise_angular
            
        return [np.clip(linear_val, 0.0, 1.0), np.clip(angular_val, -1.0, 1.0)]

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    @tf.function # Hızlandırmak için Graph modu
    def train_step(self, states, actions, rewards, next_states, dones):
        """TensorFlow Gradient Tape ile Matematiksel Eğitim"""
        
        # --- 1. CRITIC GÜNCELLEMESİ ---
        with tf.GradientTape() as tape:
            # Gelecekteki eylemi tahmin et (Target Actor)
            target_actions = self.target_actor(next_states, training=True)
            
            # Gelecekteki ödülü tahmin et (Target Critic)
            target_q_values = self.target_critic([next_states, target_actions], training=True)
            
            # Bellman Denklemi
            y = rewards + (1 - dones) * self.discount_factor * target_q_values
            
            # Şu anki Q değerini tahmin et (Main Critic)
            critic_value = self.critic_model([states, actions], training=True)
            
            # Hata hesapla
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        # Critic ağırlıklarını güncelle
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # --- 2. ACTOR GÜNCELLEMESİ ---
        with tf.GradientTape() as tape:
            # Actor yeni bir eylem seçsin
            new_actions = self.actor_model(states, training=True)
            
            # Critic bu eyleme not versin (Loss = -Q)
            critic_value = self.critic_model([states, new_actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        # Actor ağırlıklarını güncelle
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def update_target(self, target_weights, weights, tau):
        """Soft Update"""
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def train_model(self):
        # Hafızadan rastgele örnekler çek (Batch)
        batch = random.sample(self.replay_memory, self.batch_size)
        
        states = np.array([i[0][0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch]).reshape(-1, 1)
        next_states = np.array([i[3][0] for i in batch])
        dones = np.array([float(i[4]) for i in batch]).reshape(-1, 1)

        # Tensörlere çevir
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Eğitimi başlat
        self.train_step(states, actions, rewards, next_states, dones)

        # Target Modelleri Yavaşça Güncelle
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
        
        if future.result() is not None:
            state = future.result().state
            state = np.reshape(np.asarray(state), [1, self.state_size])
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

        if future.result() is not None:
            next_state = future.result().state
            next_state = np.reshape(np.asarray(next_state), [1, self.state_size])
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
