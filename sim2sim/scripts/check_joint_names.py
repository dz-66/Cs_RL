#!/usr/bin/env python3
"""
验证ROS2话题发布的关节命名是否正确
"""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

class JointNameChecker(Node):
    def __init__(self):
        super().__init__('joint_name_checker')

        # 使用BEST_EFFORT匹配发布者的QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.callback,
            qos)
        self.received = False

    def callback(self, msg):
        print("=" * 70)
        print("✅ 收到 /joint_states 消息")
        print("=" * 70)
        print(f"\n关节命名 (共{len(msg.name)}个):")
        for i, name in enumerate(msg.name):
            print(f"  [{i:2d}] {name}")

        print(f"\n关节位置:")
        for i, (name, pos) in enumerate(zip(msg.name, msg.position)):
            print(f"  {name:15s}: {pos:7.3f}")

        print(f"\n✅ 关节命名检查:")
        if 'FL_hip_joint' in msg.name[0] or 'FL_hip' in msg.name[0]:
            print("  ✅ EVA02 命名格式正确")
        elif 'FL_hip_x' in msg.name[0]:
            print("  ❌ 仍然是 Go2 命名格式")

        self.received = True

def main():
    print("启动关节命名检查器...")
    print("等待 /joint_states 消息...\n")

    rclpy.init()
    node = JointNameChecker()

    # 等待5秒接收消息
    start_time = time.time()
    while not node.received and (time.time() - start_time) < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node.received:
        print("❌ 5秒内未收到任何消息")
        print("   请确认 mujoco_ros_node 正在运行")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
