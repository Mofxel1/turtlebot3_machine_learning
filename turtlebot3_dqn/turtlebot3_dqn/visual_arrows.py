#!/usr/bin/env python3
import sys
import math
import threading

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QRectF, QPointF
from PyQt5.QtGui import QPainter, QBrush, QColor, QPolygonF, QPen

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# --- AYARLAR ---
# TurtleBot3 Waffle Pi Maksimum Hızları (Yüzdeyi hesaplamak için)
MAX_LIN_VEL = 0.26
MAX_ANG_VEL = 1.82

class ArrowWidget(QWidget):
    """
    İçi dolan ok şeklindeki özel Widget.
    """
    def __init__(self, rotation=0, parent=None):
        super().__init__(parent)
        self.value = 0.0  # 0.0 ile 1.0 arası
        self.rotation_angle = rotation
        self.setMinimumSize(100, 100)

    def set_value(self, val):
        self.value = max(0.0, min(1.0, val)) # 0-1 arasına sıkıştır
        self.update() # Yeniden çiz

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Widget'ın merkezi
        w = self.width()
        h = self.height()
        center = QPointF(w / 2, h / 2)
        size = min(w, h) * 0.8  # Kenar boşluğu bırak

        # Koordinat sistemini merkeze taşı ve döndür
        painter.translate(center)
        painter.rotate(self.rotation_angle)
        painter.translate(-center)

        # Ok Şekli (Polygon)
        # Basit bir ok çizimi: (Gövde ve Baş)
        # Koordinatlar merkeze göre ayarlanır
        arrow_width = size * 0.4
        head_width = size * 0.8
        head_len = size * 0.5
        body_len = size * 0.5

        # Okun 7 noktası
        points = [
            QPointF(center.x() - arrow_width/2, center.y() + size/2), # Sol Alt (Gövde)
            QPointF(center.x() - arrow_width/2, center.y() + size/2 - body_len), # Sol Omuz
            QPointF(center.x() - head_width/2,  center.y() + size/2 - body_len), # Sol Kanat
            QPointF(center.x(),                 center.y() - size/2), # Tepe Noktası (Uç)
            QPointF(center.x() + head_width/2,  center.y() + size/2 - body_len), # Sağ Kanat
            QPointF(center.x() + arrow_width/2, center.y() + size/2 - body_len), # Sağ Omuz
            QPointF(center.x() + arrow_width/2, center.y() + size/2)  # Sağ Alt (Gövde)
        ]
        arrow_polygon = QPolygonF(points)

        # 1. Arka Plan (Boş Ok - Gri)
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QBrush(QColor(220, 220, 220))) # Açık Gri
        painter.drawPolygon(arrow_polygon)

        # 2. Doluluk (Mavi Kısım)
        if self.value > 0:
            painter.setBrush(QBrush(QColor(0, 120, 255))) # Mavi
            
            # Maskeleme (Clipping): Sadece dolu olması gereken kısmı boya
            # Aşağıdan yukarıya doğru dolması için Rect hesapla
            fill_height = size * self.value
            fill_rect = QRectF(
                center.x() - size, 
                center.y() + size/2 - fill_height, 
                size * 2, 
                fill_height
            )
            
            # Sadece okun içinde kalan ve Rect içinde kalan alanı boya
            path = CLI_PATH = arrow_polygon.intersected(QPolygonF(fill_rect))
            painter.drawPolygon(path)

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Action Visualizer 🤖")
        self.resize(400, 400)
        self.setStyleSheet("background-color: #333; color: white;")

        layout = QGridLayout()
        self.setLayout(layout)

        # --- Okları Oluştur ---
        # 0 derece = Yukarı, 90 = Sağ, 180 = Aşağı, 270 = Sol
        self.arrow_up = ArrowWidget(rotation=0)
        self.arrow_down = ArrowWidget(rotation=180)
        self.arrow_left = ArrowWidget(rotation=270)
        self.arrow_right = ArrowWidget(rotation=90)

        # Etiketler
        label_style = "font-size: 14px; font-weight: bold; color: #EEE;"
        self.lbl_lin = QLabel("Linear: 0.00")
        self.lbl_ang = QLabel("Angular: 0.00")
        self.lbl_lin.setStyleSheet(label_style)
        self.lbl_ang.setStyleSheet(label_style)

        # --- Yerleşim (Grid) ---
        #       [   ] [UP ] [   ]
        #       [LEFT][LAB] [RIGHT]
        #       [   ] [DWN] [   ]
        
        layout.addWidget(self.arrow_up,    0, 1)
        layout.addWidget(self.arrow_left,  1, 0)
        layout.addWidget(self.lbl_lin,     1, 1, alignment=Qt.AlignCenter) # Ortada Yazı
        layout.addWidget(self.arrow_right, 1, 2)
        layout.addWidget(self.arrow_down,  2, 1)
        
        # Angular yazısını en alta koyalım
        layout.addWidget(self.lbl_ang,     3, 1, alignment=Qt.AlignCenter)

    def update_data(self, linear, angular):
        # Yazıları güncelle
        self.lbl_lin.setText(f"Hız: {linear:.2f} m/s")
        self.lbl_ang.setText(f"Dönüş: {angular:.2f} rad/s")

        # --- İLERİ / GERİ ---
        # Sigmoid (0-1) kullandığımız için robot şimdilik geri gitmez ama yine de ekleyelim.
        if linear >= 0:
            ratio = linear / MAX_LIN_VEL
            self.arrow_up.set_value(ratio)
            self.arrow_down.set_value(0)
        else:
            ratio = abs(linear) / MAX_LIN_VEL
            self.arrow_up.set_value(0)
            self.arrow_down.set_value(ratio)

        # --- SAĞ / SOL ---
        # ROS standartlarına göre: +Angular = SOL, -Angular = SAĞ
        if angular > 0: # Sola Dönüş
            ratio = abs(angular) / MAX_ANG_VEL
            self.arrow_left.set_value(ratio)
            self.arrow_right.set_value(0)
        elif angular < 0: # Sağa Dönüş
            ratio = abs(angular) / MAX_ANG_VEL
            self.arrow_left.set_value(0)
            self.arrow_right.set_value(ratio)
        else:
            self.arrow_left.set_value(0)
            self.arrow_right.set_value(0)

# --- ROS İle GUI Arasındaki Köprü ---
class RosBridge(QObject):
    data_received = pyqtSignal(float, float) # GUI'ye sinyal göndermek için

class RosSubscriber(Node):
    def __init__(self, bridge):
        super().__init__('visual_gui_node')
        self.bridge = bridge
        # Robotun aldığı son komutu dinle
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel', 
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        # Veriyi al ve GUI thread'ine gönder
        self.bridge.data_received.emit(msg.linear.x, msg.angular.z)

def run_ros(node):
    rclpy.spin(node)

def main():
    # 1. PyQt Uygulamasını Başlat
    app = QApplication(sys.argv)
    
    # 2. GUI Penceresini Oluştur
    dashboard = Dashboard()
    dashboard.show()

    # 3. ROS 2 Başlat
    rclpy.init()
    bridge = RosBridge()
    bridge.data_received.connect(dashboard.update_data) # Sinyali bağla

    ros_node = RosSubscriber(bridge)
    
    # 4. ROS'u Ayrı Bir Thread'de Çalıştır (GUI donmasın diye)
    ros_thread = threading.Thread(target=run_ros, args=(ros_node,), daemon=True)
    ros_thread.start()

    # 5. Uygulamayı Çalıştır
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
