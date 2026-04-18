#!/usr/bin/env python3
#################################################################################
# SAC LSTM - Cikarim (Inference) Modu
# Egitilmis modeli yukler, robotu hedefe yonlendirir.
# Egitim yapmaz, sadece actor agini kullanir.
#################################################################################

import os
import sys
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

import tensorflow as tf

from turtlebot3_msgs.srv import Dqn

tf.config.set_visible_devices([], 'GPU')


# ─────────────────────────────────────────────
#  AYARLAR — Buradan degistir
# ─────────────────────────────────────────────
STAGE_NUM        = 3          # Hangi stage modeli yuklensin
N_EPISODES       = 100        # Kac bolum calistirilsin (0 = sonsuz)
STEP_DELAY       = 0.05       # Adimlar arasi bekleme (saniye)

# Model dizini — egitimde kullanilan yol ile ayni olmali
HOME_DIR         = os.path.expanduser('~')
MODEL_DIR        = os.path.join(
    HOME_DIR,
    'turtlebot3_ws', 'src',
    'turtlebot3_machine_learning',
    'turtlebot3_dqn',
    'saved_model'
)
ACTOR_FILENAME   = f'stage{STAGE_NUM}_sac_actor.keras'
# ─────────────────────────────────────────────


class SACInferenceNode(Node):

    def __init__(self, stage_num: int, n_episodes: int):
        super().__init__('sac_inference')

        self.stage       = stage_num
        self.n_episodes  = n_episodes  # 0 => sonsuz

        # State boyutlari — egitimle ayni olmali
        self.stack_size    = 5
        self.raw_state_size = 74          # 72 lidar + 2 hedef
        self.state_size    = self.raw_state_size * self.stack_size  # 370

        self.stacked_state = np.zeros((1, self.state_size), dtype=np.float32)

        # ── Model yukle ──────────────────────────────────────────────────
        actor_path = os.path.join(MODEL_DIR, ACTOR_FILENAME)
        if not os.path.exists(actor_path):
            self.get_logger().error(
                f'Model dosyasi bulunamadi: {actor_path}\n'
                f'  Lutfen MODEL_DIR ve ACTOR_FILENAME ayarlarini kontrol edin.'
            )
            raise FileNotFoundError(actor_path)

        self.get_logger().info(f'Model yukleniyor: {actor_path}')
        self.actor = tf.keras.models.load_model(actor_path, compile=False)
        self.get_logger().info('Model basariyla yuklendi.')

        # ── ROS 2 servisleri ─────────────────────────────────────────────
        self.rl_agent_interface_client = self.create_client(Dqn,   'rl_agent_interface')
        self.make_environment_client   = self.create_client(Empty, 'make_environment')
        self.reset_environment_client  = self.create_client(Dqn,   'reset_environment')

        self.get_logger().info(
            f'SAC Cikarim Modu | Stage: {self.stage} | '
            f'Bolum: {"sonsuz" if n_episodes == 0 else n_episodes}'
        )

        self.run()

    # ─────────────────────────────────────────────────────────────────────
    #  Deterministik eylem secimi  (gurultu YOK, sadece ortalama kullanilir)
    # ─────────────────────────────────────────────────────────────────────
    def get_action(self, state: np.ndarray) -> list[float]:
        mean, _log_std = self.actor(state, training=False) # training=False ekledim, daha stabil olur
        action_tanh = tf.tanh(mean).numpy()[0]
        
        # CLAUDE'UN UNUTTUĞU KISIM: TurtleBot3 fiziksel hız sınırlarına ölçekleme!
        # İleri Hız: [0.0, 0.26] | Dönüş Hızı: [-1.82, 1.82]
        linear_vel = (action_tanh[0] + 1.0) * (0.26 / 2.0)
        angular_vel = action_tanh[1] * 1.82
        
        return [float(linear_vel), float(angular_vel)]

    # ─────────────────────────────────────────────────────────────────────
    #  Ortam hazirla
    # ─────────────────────────────────────────────────────────────────────
    def env_make(self):
        self.get_logger().info('Ortam hazirlaniyor...')
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('make_environment servisi bekleniyor...')
        self.make_environment_client.call_async(Empty.Request())
        time.sleep(1.5)

    # ─────────────────────────────────────────────────────────────────────
    #  Bolum basinda reset
    # ─────────────────────────────────────────────────────────────────────
    def reset_environment(self) -> np.ndarray:
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('reset_environment servisi bekleniyor...')

        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)

        self.stacked_state[:] = 0.0  # tampon sifirla

        if future.result() is not None:
            raw = np.asarray(future.result().state, dtype=np.float32)
            raw = raw.reshape(self.raw_state_size)
            for i in range(self.stack_size):
                s = i * self.raw_state_size
                self.stacked_state[0, s:s + self.raw_state_size] = raw
        else:
            self.get_logger().error('reset_environment yanit vermedii!')

        return self.stacked_state.copy()

    # ─────────────────────────────────────────────────────────────────────
    #  Tek adim
    # ─────────────────────────────────────────────────────────────────────
    def step(self, action: list[float]):
        req = Dqn.Request()
        req.action = action

        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            raw = np.asarray(future.result().state, dtype=np.float32)
            raw = raw.reshape(self.raw_state_size)

            # Kayan pencere (frame stacking)
            self.stacked_state[0, :-self.raw_state_size] = \
                self.stacked_state[0, self.raw_state_size:]
            self.stacked_state[0, -self.raw_state_size:] = raw

            return (
                self.stacked_state.copy(),
                future.result().reward,
                future.result().done,
            )
        else:
            self.get_logger().error('rl_agent_interface yanit vermedi!')
            return np.zeros((1, self.state_size)), 0.0, True

    # ─────────────────────────────────────────────────────────────────────
    #  Ana dongu
    # ─────────────────────────────────────────────────────────────────────
    def run(self):
        self.env_make()

        episode    = 0
        success_n  = 0
        fail_n     = 0

        print('\n' + '=' * 60)
        print(' SAC INFERENCE — BASLIYOR')
        print('=' * 60 + '\n')

        while True:
            episode += 1
            if self.n_episodes > 0 and episode > self.n_episodes:
                break

            state      = self.reset_environment()
            local_step = 0
            score      = 0.0

            while True:
                time.sleep(STEP_DELAY)
                local_step += 1

                action               = self.get_action(state)
                state, reward, done  = self.step(action)
                score               += reward

                if done:
                    if reward >= 50.0:
                        success_n += 1
                        sonuc = 'BASARI ✓'
                    else:
                        fail_n += 1
                        sonuc = 'BASARISIZ ✗'

                    total_done = success_n + fail_n
                    oran = 100.0 * success_n / total_done if total_done else 0.0

                    print(
                        f'Bolum: {episode:4d} | '
                        f'Skor: {score:+7.1f} | '
                        f'Adim: {local_step:4d} | '
                        f'{sonuc}  |  '
                        f'Basari: {success_n}/{total_done} ({oran:.0f}%)'
                    )
                    break

        # ── Ozet ─────────────────────────────────────────────────────────
        total = success_n + fail_n
        print('\n' + '=' * 60)
        print(f' TAMAMLANDI  —  {total} bolum')
        print(f' Basari : {success_n}  ({100*success_n/total:.1f}%)')
        print(f' Basarisiz: {fail_n}  ({100*fail_n/total:.1f}%)')
        print('=' * 60 + '\n')


# ─────────────────────────────────────────────────────────────────────────
def main(args=None):
    if args is None:
        args = sys.argv

    stage_num  = int(args[1]) if len(args) > 1 else STAGE_NUM
    n_episodes = int(args[2]) if len(args) > 2 else N_EPISODES

    rclpy.init(args=args)

    try:
        node = SACInferenceNode(stage_num, n_episodes)
        rclpy.spin(node)
    except FileNotFoundError:
        pass
    except KeyboardInterrupt:
        print('\nKullanici tarafindan durduruldu.')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
