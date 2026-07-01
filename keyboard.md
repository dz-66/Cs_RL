# 键盘控制系统开发总结

## 项目目标

为 Go2 和 EVA02 四足机器人实现基于 evdev 的键盘控制系统，通过 ROS2 与 MuJoCo 仿真通信。

---

## 遇到的问题及解决方案

### 问题 1：终端键盘输入无法与 MuJoCo 窗口兼容

**现象**：
- 最初使用 `pynput` 库进行全局键盘监听
- 在 MuJoCo viewer 窗口打开时，终端的 `input()` 或标准输入无法正常工作
- 键盘焦点在 MuJoCo 窗口和终端之间切换导致冲突

**根本原因**：
- `pynput` 依赖 X11/Wayland 窗口系统的键盘事件
- MuJoCo viewer 会拦截系统级键盘事件
- 多个应用竞争键盘焦点

**解决方案**：
使用 **evdev** 直接读取 Linux 内核级键盘事件

**优势**：
- 绕过 X11/Wayland 窗口系统
- 直接从 `/dev/input/event*` 读取硬件事件
- 不受窗口焦点影响
- 无需 MuJoCo 窗口有焦点即可工作

**使用的工具**：
```bash
pip install evdev
```

---

### 问题 2：权限不足，无法读取 `/dev/input/event*`

**现象**：
```
PermissionError: [Errno 13] Permission denied: '/dev/input/event2'
```

**根本原因**：
- `/dev/input/event*` 设备文件属于 `input` 用户组
- 普通用户默认没有读取权限

**解决方案（两步）**：

#### 步骤 1：将用户添加到 input 组
```bash
sudo usermod -a -G input $USER
```

**注意**：需要**完全重启系统**（不是注销）才能在所有 shell 中生效。

#### 步骤 2：临时权限切换（重启前使用）
在启动脚本中使用 `sg` 命令临时切换组：
```bash
sg input -c "python3 keyboard_controller_evdev.py"
```

**使用的工具**：
- `usermod` - 用户组管理
- `sg` - 临时组切换
- `groups` - 查看当前用户组

**验证方法**：
```bash
# 查看当前组
groups

# 检查权限
ls -l /dev/input/event*

# 测试读取
python3 -c "from evdev import InputDevice; print(InputDevice('/dev/input/event2'))"
```

---

### 问题 3：检测到错误的键盘设备

**现象**：
- 键盘控制器检测到 `ASUSTeK ROG OMNI RECEIVER Keyboard` (event6)
- 实际是鼠标接收器，不是真正的键盘
- 按方向键没有反应，`/cmd_vel` 一直是 0

**根本原因**：
设备检测逻辑不够精确，按照设备路径顺序匹配，鼠标接收器（event6）排在笔记本键盘（event2）前面。

**错误的代码逻辑**：
```python
# 旧逻辑：第一个包含 "keyboard" 的设备
for dev in devices:
    if 'keyboard' in dev.name.lower():
        return dev  # ROG 接收器被优先匹配
```

**解决方案**：
建立明确的优先级系统，优先匹配笔记本内置键盘

**修复后的代码**：
```python
def _find_keyboard() -> Optional[InputDevice]:
    """
    查找键盘设备
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
```

**使用的工具**：
- `evdev.list_devices()` - 列出所有输入设备
- `dev.capabilities()` - 检查设备支持的事件类型

**调试方法**：
```python
# 列出所有键盘设备及其能力
from evdev import InputDevice, list_devices, ecodes

for path in list_devices():
    dev = InputDevice(path)
    caps = dev.capabilities()
    
    if ecodes.EV_KEY in caps:
        keys = caps[ecodes.EV_KEY]
        has_arrows = all(k in keys for k in [
            ecodes.KEY_UP, ecodes.KEY_DOWN, 
            ecodes.KEY_LEFT, ecodes.KEY_RIGHT
        ])
        
        print(f"设备: {dev.name}")
        print(f"路径: {path}")
        print(f"支持方向键: {has_arrows}")
```

---

### 问题 4：机器人收到命令但不移动

**现象**：
- `/cmd_vel` 正确发布速度命令（如 `linear.x: 1.5`）
- MuJoCo 节点订阅了 `/cmd_vel`
- 但机器人在仿真中不移动或立即摔倒

**根本原因**：
RL 策略参数不匹配，模型是从网上下载的，训练时的配置未知。

**相关参数**：
- `default_dof_pos` - 默认关节角度（策略的均衡点）
- `action_scale` - 动作缩放因子
- PD 控制器增益（Kp, Kd）

**临时解决方案**：
根据机器人配置文件设置默认姿态：

**Go2**：
```python
default_dof_pos = [0.0, 1.0, -1.5] * 4  # hip_x, hip_y, knee
action_scale = 0.25
```

**EVA02**：
```python
default_dof_pos = [0.0, 0.9, -1.6] * 4  # hip, thigh, calf
action_scale = 0.25
```

**代码实现**（自动检测机器人类型）：
```python
# 根据 XML 路径自动检测
if "eva02" in xml_path.lower():
    self._default_dof_pos = np.array([
        0.0, 0.9, -1.6,   # FL
        0.0, 0.9, -1.6,   # FR
        0.0, 0.9, -1.6,   # RL
        0.0, 0.9, -1.6,   # RR
    ])
else:  # Go2
    self._default_dof_pos = np.array([
        0.0, 1.0, -1.5,   # FL
        0.0, 1.0, -1.5,   # FR
        0.0, 1.0, -1.5,   # RL
        0.0, 1.0, -1.5,   # RR
    ])
```

**使用的工具**：
- ROS2 topic 监控：`ros2 topic echo /cmd_vel`
- 关节状态检查：`ros2 topic echo /joint_states`
- 里程计验证：`ros2 topic echo /odom`

**最佳实践**：
- 找到训练配置文件（Isaac Lab / legged_gym）
- 或重新训练模型以匹配当前仿真环境
- 记录测试中稳定的参数组合

---

## 技术架构

### 系统组件

```
┌─────────────────────┐
│  笔记本键盘硬件      │
│  /dev/input/event2  │
└──────────┬──────────┘
           │ evdev 读取
           ▼
┌─────────────────────────────────┐
│  keyboard_controller_evdev.py   │
│  - 监听键盘事件                  │
│  - 映射按键 → 速度命令           │
│  - 发布 Twist 消息               │
└──────────┬──────────────────────┘
           │ ROS2 /cmd_vel
           ▼
┌─────────────────────────────────┐
│  mujoco_ros_node.py             │
│  - 订阅 /cmd_vel                │
│  - RL 策略推理                  │
│  - PD 控制器                    │
│  - MuJoCo 物理仿真              │
└─────────────────────────────────┘
```

### 按键映射

```python
KEY_MAP = {
    ecodes.KEY_UP: (1.0, 0.0, 0.0),      # ↑ 前进
    ecodes.KEY_DOWN: (-1.0, 0.0, 0.0),   # ↓ 后退
    ecodes.KEY_LEFT: (0.0, 0.0, 1.5),    # ← 左转
    ecodes.KEY_RIGHT: (0.0, 0.0, -1.5),  # → 右转
    ecodes.KEY_SPACE: (0.0, 0.0, 0.0),   # 停止
}
```

### 速度命令映射

```python
# 按键 → (vx, vy, vyaw) → Twist 消息
twist.linear.x = vx * scale_linear    # 前进/后退
twist.linear.y = vy * scale_linear    # 左移/右移（当前未使用）
twist.angular.z = vyaw * scale_angular # 转向
```

---

## 文件结构

```
sim2sim/
├── joy_ws/                              # ROS2 工作空间
│   └── src/cs_joy/
│       └── cs_joy/
│           └── keyboard_controller_evdev.py  # 键盘控制器主程序
│
├── src/ros2_bridge/
│   └── mujoco_ros_node.py              # MuJoCo + ROS2 桥接
│
├── start_keyboard.sh                    # 启动键盘控制器
├── start_mujoco.sh                      # 启动 Go2 仿真
└── start_eva02.sh                       # 启动 EVA02 仿真
```

---

## 使用方法

### 启动系统

**终端 1 - 启动 MuJoCo 仿真**：
```bash
cd /home/tino66/Cs_RL/sim2sim

# Go2 机器人
./start_mujoco.sh

# 或 EVA02 机器人
./start_eva02.sh
```

**终端 2 - 启动键盘控制器**：
```bash
cd /home/tino66/Cs_RL/sim2sim
./start_keyboard.sh
```

### 控制方式

- **↑** : 前进
- **↓** : 后退
- **←** : 左转
- **→** : 右转
- **Space** : 紧急停止

### 验证命令

```bash
# 检查节点状态
ros2 node list

# 检查话题
ros2 topic list

# 实时查看速度命令
ros2 topic echo /cmd_vel

# 查看关节状态
ros2 topic echo /joint_states

# 查看机器人位置
ros2 topic echo /odom
```

---

## 调试工具和技巧

### 1. 检查键盘设备

```bash
# 列出所有输入设备
ls -l /dev/input/event*

# 使用 evtest 测试按键（需要 sudo）
sudo evtest

# Python 脚本检查
python3 << EOF
from evdev import InputDevice, list_devices, ecodes

for path in list_devices():
    dev = InputDevice(path)
    if 'keyboard' in dev.name.lower():
        print(f"键盘: {dev.name}")
        print(f"路径: {path}")
        
        caps = dev.capabilities()
        if ecodes.EV_KEY in caps:
            keys = caps[ecodes.EV_KEY]
            print(f"支持方向键: {ecodes.KEY_UP in keys}")
EOF
```

### 2. 测试键盘事件读取

```bash
# 直接监听按键事件（需要 sg input）
sg input -c 'python3 -c "
from evdev import InputDevice, categorize, ecodes
dev = InputDevice(\"/dev/input/event2\")
print(\"监听键盘事件，按任意键测试...\")
for event in dev.read_loop():
    if event.type == ecodes.EV_KEY:
        key_event = categorize(event)
        if key_event.keystate == key_event.key_down:
            key_name = ecodes.KEY.get(key_event.scancode, \"UNKNOWN\")
            print(f\"按键: {key_name} (键码: {key_event.scancode})\")
"'
```

### 3. ROS2 话题调试

```bash
# 查看话题频率
ros2 topic hz /cmd_vel

# 查看话题信息
ros2 topic info /cmd_vel

# 查看节点信息
ros2 node info /keyboard_controller_evdev

# 手动发布测试命令
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

### 4. 查看日志

```bash
# 键盘控制器日志
tail -f /tmp/keyboard_controller.log

# MuJoCo 仿真日志
tail -f /tmp/mujoco_sim.log
```

---

## 常见问题排查

### Q: 按键没反应，`/cmd_vel` 一直是 0

**排查步骤**：
1. 检查键盘控制器是否在运行：
   ```bash
   ros2 node list | grep keyboard
   ```

2. 查看检测到的键盘设备：
   ```bash
   cat /tmp/keyboard_controller.log | grep "找到键盘"
   ```

3. 如果检测错误，重启键盘控制器：
   ```bash
   pkill -f keyboard_controller_evdev
   cd /home/tino66/Cs_RL/sim2sim
   ./start_keyboard.sh
   ```

### Q: 权限错误 `Permission denied`

**解决方案**：
```bash
# 检查是否在 input 组
groups | grep input

# 如果没有，添加并重启系统
sudo usermod -a -G input $USER
sudo reboot

# 重启前使用 sg input 临时切换
sg input -c './start_keyboard.sh'
```

### Q: 机器人收到命令但不稳定/摔倒

**可能原因**：
- RL 策略参数不匹配
- `action_scale` 太大或太小
- `default_dof_pos` 与训练时不一致

**调试方法**：
```bash
# 尝试更小的 action_scale
python -m src.ros2_bridge.mujoco_ros_node \
    --model model/policy.onnx \
    --action-scale 0.15

# 查看机器人实时状态
ros2 topic echo /odom --field pose.pose.position
```

---

## 关键经验总结

### 1. evdev vs pynput

| 特性 | evdev | pynput |
|------|-------|--------|
| 工作层级 | 内核 /dev/input | X11/Wayland |
| 需要权限 | input 组 | 无特殊要求 |
| 窗口焦点 | 不受影响 | 需要焦点 |
| 与 MuJoCo 兼容 | ✅ 完美 | ❌ 冲突 |
| 跨平台 | ❌ Linux only | ✅ 跨平台 |

**结论**：对于 Linux + MuJoCo 场景，evdev 是最佳选择。

### 2. 权限管理

- **永久方案**：`usermod -a -G input` + 重启
- **临时方案**：`sg input -c` 包裹命令
- **验证方法**：直接读取 `/dev/input/event*` 测试

### 3. 设备检测策略

- 明确优先级：笔记本键盘 > 外接键盘 > 通用键盘设备
- 检查设备能力（方向键支持）
- 记录检测到的设备名称便于调试

### 4. ROS2 开发最佳实践

- 使用 `--symlink-install` 加快迭代
- 分离日志文件便于调试
- 使用 `ros2 topic echo` 验证通信
- 后台运行时重定向输出到日志

---

## 性能指标

- **键盘采样率**：evdev 读取循环（~1000Hz）
- **ROS2 发布频率**：20Hz（50ms 定时器）
- **MuJoCo 仿真频率**：~500Hz（2ms 步长）
- **端到端延迟**：< 100ms（按键 → 机器人响应）

---

## 未来改进方向

### 短期
- [ ] 优化设备检测逻辑（支持更多键盘类型）
- [ ] 添加配置文件（自定义按键映射）
- [ ] 实现速度渐变（避免突变）

### 长期
- [ ] 支持游戏手柄（通过 evdev 读取 joystick）
- [ ] 添加键盘 LED 反馈（状态指示）
- [ ] 实现录制/回放功能
- [ ] GUI 控制面板

---

## 参考资料

- [evdev Python 文档](https://python-evdev.readthedocs.io/)
- [ROS2 Humble 文档](https://docs.ros.org/en/humble/)
- [MuJoCo 文档](https://mujoco.readthedocs.io/)
- [Linux input 子系统](https://www.kernel.org/doc/html/latest/input/)

---

## 贡献者

- 键盘控制系统设计与实现
- 问题诊断与调试
- 文档编写

**最后更新**：2025-07-01
