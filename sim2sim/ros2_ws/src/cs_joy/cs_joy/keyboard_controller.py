"""
键盘控制 → ROS2 Twist 发布节点
通过键盘输入直接发布 /cmd_vel 指令

用法:
    ros2 run cs_joy keyboard_controller
    
    # 或直接运行
    python -m cs_joy.keyboard_controller
"""
import sys
import os
import select
import termios
import tty
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


# 键盘映射
KEY_MAP = {
    "w": (1.0, 0.0, 0.0),     # 前进
    "s": (-1.0, 0.0, 0.0),    # 后退
    "a": (0.0, 0.5, 0.0),     # 左移
    "d": (0.0, -0.5, 0.0),    # 右移
    "q": (0.0, 0.0, 1.5),     # 左转
    "e": (0.0, 0.0, -1.5),    # 右转
    " ": (0.0, 0.0, 0.0),     # 停止
}

HELP_TEXT = """
========================================
  Go2 键盘控制器
========================================
  W/S  : 前进 / 后退
  A/D  : 左移 / 右移
  Q/E  : 左转 / 右转
  Space: 紧急停止
  Ctrl+C: 退出
========================================
"""


class KeyboardController(Node):
    """键盘 → Twist 发布节点"""
    
    def __init__(self):
        super().__init__("keyboard_controller")
        self._publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self._timer = self.create_timer(0.05, self._publish_cmd)  # 20Hz
        
        self._current_keys = set()
        self._running = True
        
        # 速度缩放
        self._scale_linear = 1.5   # m/s
        self._scale_angular = 2.0  # rad/s
        
        self.get_logger().info("键盘控制器已启动")
        print(HELP_TEXT)
    
    def add_key(self, key: str):
        self._current_keys.add(key)
    
    def remove_key(self, key: str):
        self._current_keys.discard(key)
    
    def _publish_cmd(self):
        """发布当前按键对应的 Twist"""
        twist = Twist()
        
        for key in self._current_keys:
            if key in KEY_MAP:
                vx, vy, vyaw = KEY_MAP[key]
                twist.linear.x += vx * self._scale_linear
                twist.linear.y += vy * self._scale_linear
                twist.angular.z += vyaw * self._scale_angular
        
        # 限幅
        twist.linear.x = max(-self._scale_linear, 
                              min(self._scale_linear, twist.linear.x))
        twist.linear.y = max(-self._scale_linear, 
                              min(self._scale_linear, twist.linear.y))
        twist.angular.z = max(-self._scale_angular, 
                               min(self._scale_angular, twist.angular.z))
        
        self._publisher.publish(twist)
    
    def stop(self):
        self._running = False
        # 发送零速度
        self._publisher.publish(Twist())


def get_key(settings) -> Optional[str]:
    """非阻塞读取按键"""
    timeout = 0.1
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if r:
        return sys.stdin.read(1)
    return None


def keyboard_loop():
    """键盘监听线程"""
    # 保存终端设置
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while True:
            key = get_key(old_settings)
            if key:
                key_lower = key.lower()
                
                if key == "\x03":  # Ctrl+C
                    break
                
                if key_lower in KEY_MAP:
                    yield ("press", key_lower)
                elif key == "\x7f":  # Backspace
                    pass
            
            yield ("tick", None)
    
    finally:
        termios.tcsetattr(sys.stdin, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardController()
    
    # 键盘输入线程
    def input_thread():
        for event, key in keyboard_loop():
            if event == "press":
                node.add_key(key)
            elif event == "tick":
                # 清除所有按键 (需要持续按住)
                node._current_keys.clear()
            
            if not node._running:
                break
    
    input_t = threading.Thread(target=input_thread, daemon=True)
    input_t.start()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        print("\n键盘控制器已退出")


if __name__ == "__main__":
    main()
