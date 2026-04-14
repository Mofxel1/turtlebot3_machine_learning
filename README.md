# TurtleBot3
<img src="https://raw.githubusercontent.com/ROBOTIS-GIT/emanual/master/assets/images/platform/turtlebot3/logo_turtlebot3.png" width="300">

- Active Branches: noetic, humble, jazzy, main(rolling)
- Legacy Branches: *-devel

## Open Source Projects Related to TurtleBot3
- [turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3)
- [turtlebot3_msgs](https://github.com/ROBOTIS-GIT/turtlebot3_msgs)
- [turtlebot3_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations)
- [turtlebot3_manipulation](https://github.com/ROBOTIS-GIT/turtlebot3_manipulation)
- [turtlebot3_manipulation_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_manipulation_simulations)
- [turtlebot3_applications](https://github.com/ROBOTIS-GIT/turtlebot3_applications)
- [turtlebot3_applications_msgs](https://github.com/ROBOTIS-GIT/turtlebot3_applications_msgs)
- [turtlebot3_machine_learning](https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning)
- [turtlebot3_autorace](https://github.com/ROBOTIS-GIT/turtlebot3_autorace)
- [turtlebot3_home_service_challenge](https://github.com/ROBOTIS-GIT/turtlebot3_home_service_challenge)
- [hls_lfcd_lds_driver](https://github.com/ROBOTIS-GIT/hls_lfcd_lds_driver)
- [ld08_driver](https://github.com/ROBOTIS-GIT/ld08_driver)
- [open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator)
- [dynamixel_sdk](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- [OpenCR-Hardware](https://github.com/ROBOTIS-GIT/OpenCR-Hardware)
- [OpenCR](https://github.com/ROBOTIS-GIT/OpenCR)

## Documentation, Videos, and Community

### Official Documentation
- ⚙️ **[ROBOTIS DYNAMIXEL](https://dynamixel.com/)**
- 📚 **[ROBOTIS e-Manual for Dynamixel SDK](http://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/)**
- 📚 **[ROBOTIS e-Manual for TurtleBot3](http://turtlebot3.robotis.com/)**
- 📚 **[ROBOTIS e-Manual for OpenMANIPULATOR-X](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/)**

### Learning Resources
- 🎥 **[ROBOTIS YouTube Channel](https://www.youtube.com/@ROBOTISCHANNEL)**
- 🎥 **[ROBOTIS Open Source YouTube Channel](https://www.youtube.com/@ROBOTISOpenSourceTeam)**
- 🎥 **[ROBOTIS TurtleBot3 YouTube Playlist](https://www.youtube.com/playlist?list=PLRG6WP3c31_XI3wlvHlx2Mp8BYqgqDURU)**
- 🎥 **[ROBOTIS OpenMANIPULATOR YouTube Playlist](https://www.youtube.com/playlist?list=PLRG6WP3c31_WpEsB6_Rdt3KhiopXQlUkb)**

### Community & Support
- 💬 **[ROBOTIS Community Forum](https://forum.robotis.com/)**
- 💬 **[TurtleBot category from ROS Community](https://discourse.ros.org/c/turtlebot/)**



# 🏎️ TurtleBot3 Otonom Ralli Pilotu (SAC + LSTM)

Bu proje, TurtleBot3 (Waffle Pi) robotunun engellerden kaçarak hedefe ulaşmasını sağlayan Continuous Control (Sürekli Kontrol) tabanlı bir Derin Pekiştirmeli Öğrenme (Deep RL) mimarisidir. 
Endüstri standardı **Soft Actor-Critic (SAC)** ve zaman/hafıza ilişkisi için **LSTM** sinir ağları kullanılmıştır.

---

## 🚀 Sistemi Ayağa Kaldırma (5 Kutsal Terminal)

Sistemi başlatmak için 4 ayrı terminal açın ve aşağıdaki komutları sırasıyla çalıştırın:

### 1. Fizik Motoru ve Simülasyon (Gazebo)
Robotun içinde eğitileceği fiziksel dünyayı (Aşama 3) başlatır.
```bash
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage3.launch.py

2. Hakem ve Kurallar (Environment)

Robotun ödül ve cezalarını hesaplayan, engelleri ve hedefi belirleyen çevreyi başlatır.
Bash

ros2 run turtlebot3_dqn dqn_environment

3. Yapay Zeka Beyni (SAC + LSTM Ajanı)

Robotun kontrolünü sağlayan ve öğrenme matrislerini güncelleyen ana sinir ağını başlatır. (Sondaki '3' rakamı, 3. Aşama haritası içindir).
Bash

ros2 run turtlebot3_dqn sac_lstm_agent 3

4. Hareketli Engellerin Kontrolü (Gazebo Node)

Aşama 3'teki hareketli sütunlara hız komutları göndererek ortamı dinamik ve zorlu hale getirir. (Sondaki '3' rakamı, 3. Aşama haritası içindir).
Bash

ros2 run turtlebot3_dqn dqn_gazebo 3

5. Telemetri ve Analiz (TensorBoard)

Yapay zekanın eğitim durumunu, Alpha (Merak) sönümlenmesini ve toplam skor grafiklerini anlık olarak tarayıcıdan izlemek içindir.
Bash

tensorboard --logdir ~/turtlebot3_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/saved_model/logs_sac

(Çalıştırdıktan sonra tarayıcınızda http://localhost:6006 adresine gidin).
```

🛠️ Geliştirici İpuçları

    Hızlı Derleme Kısayolu: Python (.py) dosyalarında yaptığınız değişikliklerin sistemi her seferinde yeniden derlemeden anında uygulanması için projeyi aşağıdaki bayrakla derleyin:
```bash

    colcon build --packages-select turtlebot3_dqn --symlink-install

    Simülasyonu Hızlandırmak: Gazebo açıldığında sol panelden Physics menüsüne gidin ve real_time_update_rate değerini 0 yapın. (Not: Fizik hataları başlarsa bilgisayar hızınıza göre optimize edin).

    Model Kayıtları: Eğitilen modelin ağırlıkları (.keras formatında) her 10 bölümde bir otomatik olarak saved_model klasörüne kaydedilir.
```
## 🚀 Gelecek Planları ve Hiperparametre Optimizasyonu

Projenin bir sonraki fazında, ajanın şoförlük kabiliyetlerini ve ödül mekanizmasını stabilize etmek adına aşağıdaki ileri düzey tekniklerin entegrasyonu planlanmaktadır:

### 1. Otomatik Ödül Şekillendirme (Automated Reward Shaping)
Ajanın "korkaklık" (donut çizme) veya "aşırı özgüven" gibi yan yollara sapmasını engellemek için ödül katsayıları (Zaman cezası, Hedef ödülü, Engel riski) **Optuna** kütüphanesi kullanılarak Bayesian Optimizasyon yöntemiyle otomatik olarak belirlenecektir.

### 2. PBT (Population Based Training) Entegrasyonu
DeepMind tarafından popüler hale getirilen PBT metodu ile:
* Aynı anda farklı ödül katsayılarına sahip birden fazla ajan paralel simülasyonlarda yarıştırılacaktır.
* Başarılı olan ajanların ağırlıkları ve ödül parametreleri (mutasyona uğratılarak) diğerlerine aktarılacaktır.
* Bu sayede dinamik engellere (Stage 3) karşı en dayanıklı "süper ajan" evrimsel süreçle seçilecektir.

### 3. Paralel Simülasyon Yönetimi (Distributed Training)
Eğitim süresini minimize etmek adına **Ray Tune** veya **Docker Container** altyapısı kullanılarak birden fazla Gazebo instance'ının aynı anda çalıştırılması ve merkezi bir Learner üzerinden verilerin toplanması hedeflenmektedir.
