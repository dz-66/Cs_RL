"""
手柄 ↔ ROS2 Twist 桥接节点
将手柄输入 (joy_msg) 转换为 /cmd_vel (Twist)

原理:
  左摇杆上下 → 前进/后退 (linear.x)
  左摇杆左右 → 左移/右移 (linear.y)  
  右摇杆左右 → 左转/右转 (angular.z)

用法:
    ros2 run cs_joy joy_controller

依赖:
    ros2 launch cs_joy joy.launch.py  # 先启用手柄驱动
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class JoyController(Node):
    """手柄 → Twist 转换节点"""
    
    def __init__(self):
        super().__init__("joy_controller")
        
        # 订阅手柄
        self._joy_sub = self.create_subscription(
            Joy, "/joy", self._joy_callback, 10
        )
        
        # 发布 Twist
        self._twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # 手柄映射配置 (Xbox 默认)
        # 轴:
        #   0: 左摇杆水平
        #   1: 左摇杆垂直
        #   3: 右摇杆水平
        #   4: 右摇杆垂直
        self._axis_map = {
            "linear_x": 1,      # 左摇杆上下 → 前后
            "linear_y": 0,      # 左摇杆左右 → 左右
            "angular_z": 3,     # 右摇杆左右 → 转向
        }
        
        # 速度缩放
        self._max_linear_x = 1.5    # m/s
        self._max_linear_y = 0.8    # m/s
        self._max_angular_z = 2.0   # rad/s
        
        # 死区
        self._deadzone = 0.1
        
        # 使能按钮 (按住 LB 才发送指令)
        self._enable_button = 4  # LB
        self._enabled = False
        
        # 步态切换按钮
        self._gait_buttons = {
            0: "trot",      # A
            1: "stand",     # B
            2: "pace",      # X
            3: "bound",     # Y
        }
        
        self.get_logger().info("手柄控制器已启动，按住 LB 按钮开始控制")
    
    def _apply_deadzone(self, value: float) -> float:
        """死区处理"""
        if abs(value) < self._deadzone:
            return 0.0
        # 重新映射: [deadzone, 1.0] → [0.0, 1.0]
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self._deadzone) / (1.0 - self._deadzone)
    
    def _joy_callback(self, msg: Joy):
        """处理手柄输入"""
        # 检查使能按钮
        self._enabled = (
            len(msg.buttons) > self._enable_button
            and msg.buttons[self._enable_button] == 1
        )
        
        twist = Twist()
        
        if self._enabled:
            # 读取轴值
            ax = msg.axes
            
            # 线速度
            if len(ax) > self._axis_map["linear_x"]:
                raw = -ax[self._axis_map["linear_x"]]  # Xbox: 推上为负
                twist.linear.x = self._apply_deadzone(raw) * self._max_linear_x
            
            if len(ax) > self._axis_map["linear_y"]:
                raw = ax[self._axis_map["linear_y"]]
                twist.linear.y = self._apply_deadzone(raw) * self._max_linear_y
            
            # 角速度
            if len(ax) > self._axis_map["angular_z"]:
                raw = ax[self._axis_map["angular_z"]]
                twist.angular.z = self._apply_deadzone(raw) * self._max_angular_z
            
            self.get_logger().debug(
                f"CMD: vx={twist.linear.x:.2f}, vy={twist.linear.y:.2f}, "
                f"vyaw={twist.angular.z:.2f}"
            )
        
        self._twist_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = JoyController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
