#!/usr/bin/env python3
import collections, os, random, sys, time, termios, tty, select
import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, LayerNormalization, LeakyReLU, LSTM, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.config.set_visible_devices([], 'GPU')
settings = termios.tcgetattr(sys.stdin)

def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class SACLSTMTeleopAgent(Node):
    def __init__(self, stage_num, max_ep):
        super().__init__('sac_lstm_teleop_agent')
        self.stage = int(stage_num)
        self.autonomous_mode = False
        self.raw_state_size, self.stack_size = 26, 3
        self.state_size = self.raw_state_size * self.stack_size
        self.replay_memory = collections.deque(maxlen=100000)
        self.batch_size = 64
        
        self.gamma, self.tau = 0.99, 0.005
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.target_entropy = -2.0
        
        self.actor = self.build_model('actor')
        self.critic_1 = self.build_model('critic')
        self.critic_2 = self.build_model('critic')
        self.target_critic_1 = self.build_model('critic')
        self.target_critic_2 = self.build_model('critic')
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_opt = Adam(0.0003); self.critic_1_opt = Adam(0.0003); self.critic_2_opt = Adam(0.0003); self.alpha_opt = Adam(0.00003)

        self.rl_client = self.create_client(Dqn, 'rl_agent_interface')
        self.reset_client = self.create_client(Dqn, 'reset_environment')
        self.make_client = self.create_client(Empty, 'make_environment')
        
        self.process()

    def build_model(self, type):
        state_in = Input(shape=(self.state_size,))
        x = Reshape((self.stack_size, self.raw_state_size))(state_in)
        # LSTM boyutu 256'dan 128'e düşürüldü (Frame stacking ile örtüşme engellendi)
        x = LSTM(128, return_sequences=False)(x)
        x = LayerNormalization()(x); x = LeakyReLU()(x)
        x = Dense(128)(x); x = LayerNormalization()(x); x = LeakyReLU()(x)
        if type == 'actor':
            mean = Dense(2, activation='linear')(x)
            log_std = Dense(2, activation='linear')(x)
            return Model(state_in, [mean, log_std])
        action_in = Input(shape=(2,))
        x = Concatenate()([x, action_in])
        x = Dense(256)(x); x = LeakyReLU()(x)
        return Model([state_in, action_in], Dense(1, activation='linear')(x))

    def get_action(self, state):
        mean, log_std = self.actor(state)
        log_std = tf.clip_by_value(log_std, -20.0, 2.0)
        std = tf.exp(log_std)
        if not self.autonomous_mode:
            action = tf.tanh(mean)
        else:
            noise = tf.random.normal(shape=mean.shape)
            sampled_action = mean + std * noise
            action = tf.tanh(sampled_action)
            
        action = action.numpy()[0]
        # [0, 1] ÇEVRİMİ TAMAMEN KALDIRILDI! Ajan doğrudan [-1, 1] döndürür.
        linear_val = np.clip(action[0], -1.0, 1.0) 
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

    def update_target_networks(self):
        for target_weight, weight in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))
        for target_weight, weight in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def train_model(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        states = np.array([i[0][0] for i in batch], dtype=np.float32)
        actions = np.array([i[1] for i in batch], dtype=np.float32) # [0, 1] KİRLİLİĞİ KALDIRILDI!
        rewards = np.array([i[2] for i in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([i[3][0] for i in batch], dtype=np.float32)
        dones = np.array([float(i[4]) for i in batch], dtype=np.float32).reshape(-1, 1)

        self.train_step(tf.convert_to_tensor(states), tf.convert_to_tensor(actions), tf.convert_to_tensor(rewards), tf.convert_to_tensor(next_states), tf.convert_to_tensor(dones))

    def process(self):
        while not self.make_client.wait_for_service(timeout_sec=1.0): pass
        self.make_client.call_async(Empty.Request())
        
        for ep in range(1, 5001):
            future = self.reset_client.call_async(Dqn.Request())
            rclpy.spin_until_future_complete(self, future)
            raw_s = np.array(future.result().state)
            state = np.tile(raw_s, (1, self.stack_size))
            score = 0
            
            print(f"\n--- Bölüm {ep} Başladı. Mod: {'OTONOM' if self.autonomous_mode else 'MANUEL'} ---")
            
            while True:
                key = get_key()
                if key == 'o' and not self.autonomous_mode:
                    self.autonomous_mode = True
                    print("\n🧠 ŞOK EĞİTİM BAŞLADI (Sürüşün taklit ediliyor)...")
                    for _ in range(1000): 
                        if len(self.replay_memory) > 64: self.train_model()
                    print("🚀 OTONOM MOD AKTİF!")
                
                if not self.autonomous_mode:
                    lin, ang = 0.0, 0.0
                    if key == 'w': lin = 0.8
                    elif key == 's': lin = -0.8
                    elif key == 'a': ang = 0.8
                    elif key == 'd': ang = -0.8
                    action = [lin, ang]
                else:
                    action = self.get_action(state)

                req = Dqn.Request(); req.action = [float(action[0]), float(action[1])]
                future = self.rl_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                
                res = future.result()
                next_raw_s = np.array(res.state)
                next_state = np.append(state[:, self.raw_state_size:], [next_raw_s], axis=1)
                
                self.replay_memory.append((state, action, res.reward, next_state, res.done))
                if self.autonomous_mode and len(self.replay_memory) > 64: self.train_model()
                
                state, score = next_state, score + res.reward
                if res.done: break
            print(f"Skor: {score:.2f} | Hafıza: {len(self.replay_memory)}")

def main(args=None):
    rclpy.init(args=args); node = SACLSTMTeleopAgent(sys.argv[1], 5000); rclpy.spin(node)

if __name__ == '__main__': main()
