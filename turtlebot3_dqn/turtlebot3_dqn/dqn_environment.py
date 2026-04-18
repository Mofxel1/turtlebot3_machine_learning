#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
# Orhan tarafindan SAC Algoritmasina ozel olarak modifiye edilmistir.
#################################################################################

import math
import os

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal


ROS_DISTRO = os.environ.get('ROS_DISTRO')


class RLEnvironment(Node):

    def __init__(self):
        super().__init__('rl_environment')
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 5
        self.max_step = 800

        self.done = False
        self.fail = False
        self.succeed = False

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.scan_ranges = []
        self.front_ranges = []
        self.min_obstacle_distance = 10.0
        self.is_front_min_actual_front = False

        self.local_step = 0
        self.last_angular_action = 0.0  
        self.stop_cmd_vel_timer = None
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]
        
        # YENİ: Dinamik engel takip değişkeni
        self.prev_min_obstacle_dist = 3.5

        qos = QoSProfile(depth=10)

        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_sub_callback, qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_sub_callback, qos_profile_sensor_data
        )

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal, 'task_succeed', callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal, 'task_failed', callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal, 'initialize_env', callback_group=self.clients_callback_group
        )

        self.rl_agent_interface_service = self.create_service(
            Dqn, 'rl_agent_interface', self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty, 'make_environment', self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn, 'reset_environment', self.reset_environment_callback
        )

    def make_environment_callback(self, request, response):
        self.get_logger().info('Make environment called')
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for initialize the environment is not available, waiting ...')
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        response_goal = future.result()
        if not response_goal.success:
            self.get_logger().error('initialize environment request failed')
        else:
            self.get_logger().info('goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y))
        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.init_goal_distance = self.goal_distance
        self.prev_goal_distance = self.init_goal_distance
        self.prev_min_obstacle_dist = min(self.scan_ranges) if self.scan_ranges else 3.5
        
        response.state = state
        self.done = False
        self.fail = False
        self.succeed = False
        self.local_step = 0  
        return response

    def call_task_succeed(self):
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task succeed is not available, waiting ...')
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task succeed finished')
        else:
            self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task failed is not available, waiting ...')
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task failed finished')
        else:
            self.get_logger().error('task failed service call failed')

    def scan_sub_callback(self, scan):
        self.scan_ranges = []
        self.front_ranges = []
        self.front_angles = []

        num_of_lidar_rays = len(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        if num_of_lidar_rays > 0:
            self.front_distance = scan.ranges[0]
        else:
            self.front_distance = 0.0

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = scan.ranges[i]

            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)
            self.front_ranges.append(distance)
            self.front_angles.append(angle)

        self.min_obstacle_distance = min(self.scan_ranges) if self.scan_ranges else 10.0
        self.front_min_obstacle_distance = min(self.front_ranges) if self.front_ranges else 10.0

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state(self):
        state = []
        
        # YENİ: STATE NORMALİZASYONU
        norm_goal_dist = min(self.goal_distance / 3.5, 1.0)
        norm_goal_angle = self.goal_angle / math.pi
        
        state.append(float(norm_goal_dist))
        state.append(float(norm_goal_angle))
        
        # Lidar donanımında 72 atım ayarladığımız için burası direkt 72 boyutlu olacak
        for var in self.front_ranges:
            state.append(float(min(var / 3.5, 1.0)))
        
        self.local_step += 1

        if self.goal_distance < 0.20:
            self.get_logger().info('Goal Reached')
            self.succeed = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.call_task_succeed()

        if self.min_obstacle_distance < 0.15:
            self.get_logger().info('Collision happened')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.call_task_failed()

        if self.local_step >= self.max_step:
            self.get_logger().info('Time out!')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.call_task_failed()

        return state

    def calculate_reward(self):
        # 1. Yonelme Odulu - Aralik: [-1.0, +1.0]
        yaw_reward = 1.0 - (2.0 * abs(self.goal_angle) / math.pi)

        # 2. Mesafe Odulu - DUZELTME: carpan 25.0 -> 15.0
        distance_reward = (self.prev_goal_distance - self.goal_distance) * 15.0
        self.prev_goal_distance = self.goal_distance

        # 3. Zaman Cezasi - DUZELTME: -0.7 -> -0.2
        # Eski deger 800 adimda -560 yapiyordu, erken crash'i ozendiriyordu
        time_penalty = -0.2

        # 4. Engel Cezasi - DUZELTME: safe_dist 0.70 -> 0.50
        obstacle_reward = 0.0
        safe_dist = 0.50

        if self.min_obstacle_distance < safe_dist:
            risk = (safe_dist - self.min_obstacle_distance) / safe_dist
            obstacle_reward = -5.0 * (risk ** 2)

        step_reward = yaw_reward + distance_reward + obstacle_reward + time_penalty

        print(f"Yaw: {yaw_reward:+.2f} | Dist: {distance_reward:+.2f} | Obs: {obstacle_reward:+.2f} | Time: {time_penalty:+.2f} || Adim Skoru: {step_reward:+.2f}")

        # --- FINAL HUKMU --- DUZELTME: +-200 -> +-100 (olcek normalize)
        if self.succeed:
            reward = 100.0
        elif self.fail:
            reward = -100.0
        else:
            reward = step_reward

        return reward

    def rl_agent_interface_callback(self, request, response):
        action_list = request.action
        
        MAX_LIN_VEL = 0.26
        MAX_ANG_VEL = 1.82

        # Ajanın gönderdiği [-1, 1] aralığını matematiksel olarak [0, 1] aralığına çekiyoruz.
        normalized_linear = (float(action_list[0]) + 1.0) / 2.0 
        linear_cmd = normalized_linear * MAX_LIN_VEL

        angular_cmd = float(action_list[1]) * MAX_ANG_VEL

        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = linear_cmd
            msg.angular.z = angular_cmd
        else:
            msg = TwistStamped()  # DÜZELTME: msg tanımlanmadan kullanılıyordu
            msg.twist.linear.x = linear_cmd
            msg.twist.angular.z = angular_cmd

        self.cmd_vel_pub.publish(msg)

        if self.stop_cmd_vel_timer is None:
            self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)

        response.state = self.calculate_state()
        response.reward = self.calculate_reward()
        response.done = self.done

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    rl_environment = RLEnvironment()
    try:
        while rclpy.ok():
            rclpy.spin_once(rl_environment, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rl_environment.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
