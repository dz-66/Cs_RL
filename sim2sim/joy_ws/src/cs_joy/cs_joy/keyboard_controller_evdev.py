"""
键盘控制 → ROS2 Twist 发布节点 (evdev 版本)
通过 Linux evdev 直接读取内核级键盘事件

特性:
  - 绕过终端/Wayland/X11，直接读 /dev/input/event*
  - 支持按键按下/释放状态追踪
  - 无需窗口焦点

依赖:
    sudo apt install python3-evdev
    # 或 pip install evdev

权限:
    需要读取 /dev/input/event*，两种方案:
    1. 添加用户到 input 组: sudo usermod -a -G input $USER (需重新登录)
    2. 临时授权: sudo chmod +r /dev/input/event*

用法:
    ros2 run cs_joy keyboard_controller_evdev

    # 或直接运行
    python -m cs_joy.keyboard_controller_evdev
"""
import sys
import time
import threading
from typing import Optional, Set

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    from evdev import InputDevice, categorize, ecodes, list_devices
    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False
    print("[ERROR] evdev 未安装，请运行: sudo apt install python3-evdev")
    sys.exit(1)


# 键盘码映射 (evdev scancodes)
KEY_MAP = {
    ecodes.KEY_UP: (1.0, 0.0, 0.0),      # ↑ 前进
    ecodes.KEY_DOWN: (-1.0, 0.0, 0.0),   # ↓ 后退
    ecodes.KEY_LEFT: (0.0, 0.0, 1.5),    # ← 左转
    ecodes.KEY_RIGHT: (0.0, 0.0, -1.5),  # → 右转
    ecodes.KEY_SPACE: (0.0, 0.0, 0.0),   # 停止
}

HELP_TEXT = """
========================================
  Go2 键盘控制器 (evdev)
========================================
  ↑/↓       : 前进 / 后退
  ←/→       : 左转 / 右转
  Space     : 紧急停止
  Ctrl+C    : 退出
========================================
"""


def find_keyboard_device() -> Optional[InputDevice]:
    """
    自动查找键盘设备

    优先级:
      1. AT Translated Set 2 keyboard (笔记本内置键盘)
      2. 包含 "keyboard" 名称的设备
      3. 支持方向键的设备
    """
    devices = [InputDevice(path) for path in list_devices()]

    # 1. 优先: AT Translated Set 2 keyboard (笔记本内置键盘)
    for dev in devices:
        if 'AT Translated Set 2 keyboard' in dev.name:
            print(f"[evdev] 找到键盘: {dev.name} ({dev.path})")
            return dev

    # 2. 其他包含 keyboard 的设备
    for dev in devices:
        if 'kbd' in dev.path or 'keyboard' in dev.name.lower():
            print(f"[evdev] 找到键盘: {dev.name} ({dev.path})")
            return dev

    # 3. 回退: 检查是否支持方向键
    for dev in devices:
        caps = dev.capabilities(verbose=False)
        if ecodes.EV_KEY in caps:
            keys = caps[ecodes.EV_KEY]
            if ecodes.KEY_UP in keys and ecodes.KEY_DOWN in keys:
                print(f"[evdev] 找到键盘 (回退): {dev.name} ({dev.path})")
                return dev

    return None


class KeyboardControllerEvdev(Node):
    """键盘 → Twist 发布节点 (evdev)"""

    def __init__(self):
        super().__init__("keyboard_controller_evdev")
        self._publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self._timer = self.create_timer(0.05, self._publish_cmd)  # 20Hz

        self._pressed_keys: Set[int] = set()
        self._key_lock = threading.Lock()
        self._running = True

        # 速度缩放
        self._scale_linear = 1.5   # m/s
        self._scale_angular = 2.0  # rad/s

        # 查找键盘设备
        self._device = find_keyboard_device()
        if self._device is None:
            self.get_logger().error("未找到键盘设备!")
            self.get_logger().error("请检查权限: sudo usermod -a -G input $USER")
            raise RuntimeError("键盘设备不可用")

        # 启动事件监听线程
        self._listen_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._listen_thread.start()

        self.get_logger().info(f"键盘控制器已启动 (设备: {self._device.name})")
        print(HELP_TEXT)

    def _event_loop(self):
        """evdev 事件监听循环"""
        try:
            for event in self._device.read_loop():
                if not self._running:
                    break

                # 只处理按键事件
                if event.type == ecodes.EV_KEY:
                    key_event = categorize(event)

                    # 按下
                    if key_event.keystate == key_event.key_down:
                        with self._key_lock:
                            if key_event.scancode in KEY_MAP:
                                self._pressed_keys.add(key_event.scancode)

                    # 释放
                    elif key_event.keystate == key_event.key_up:
                        with self._key_lock:
                            self._pressed_keys.discard(key_event.scancode)

        except OSError as e:
            if self._running:
                self.get_logger().error(f"设备读取错误: {e}")
                self.get_logger().error("可能需要权限: sudo usermod -a -G input $USER")

    def _publish_cmd(self):
        """发布当前按键对应的 Twist"""
        twist = Twist()

        with self._key_lock:
            for key_code in self._pressed_keys:
                if key_code in KEY_MAP:
                    vx, vy, vyaw = KEY_MAP[key_code]
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
        """停止节点"""
        self._running = False
        # 发送零速度
        self._publisher.publish(Twist())
        if self._device:
            self._device.close()


def main(args=None):
    if not HAS_EVDEV:
        print("[ERROR] evdev 未安装")
        print("安装方法:")
        print("  sudo apt install python3-evdev")
        print("  # 或")
        print("  pip install evdev")
        sys.exit(1)

    rclpy.init(args=args)

    try:
        node = KeyboardControllerEvdev()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        rclpy.shutdown()
        sys.exit(1)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        print("\n键盘控制器已退出")


if __name__ == "__main__":
    main()
