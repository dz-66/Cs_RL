"""
ROS2 ↔ MuJoCo 桥接节点
接收手柄/键盘的 Twist 指令，控制 MuJoCo 中的四足机器人

架构:
  ┌──────────┐    /cmd_vel     ┌────────────────┐    PD控制    ┌──────────┐
  │ 手柄节点  │ ──────────────→ │ mujoco_ros_node│ ──────────→ │  MuJoCo  │
  │(joy→Twist)│   Twist msg    │ (桥接节点)      │  关节指令    │  仿真器   │
  └──────────┘                 │                 │ ←────────── │          │
                               │ 发布: /joint_states        │  └──────────┘
  ┌──────────┐    /cmd_vel     │  /odom                     │
  │ 键盘节点  │ ──────────────→ │  /foot_contacts            │
  └──────────┘                 └────────────────────────────┘

用法:
    # 仅仿真 (无 ROS)
    python -m src.ros2_bridge.mujoco_ros_node --standalone
    
    # 作为 ROS2 节点运行
    ros2 run cs_joy mujoco_ros_node
"""
import os
import sys
import time
import math
import argparse
import threading
import numpy as np
from pathlib import Path
from typing import Optional
from collections import deque

import mujoco
import mujoco.viewer
from mujoco import MjModel, MjData

# ROS2 依赖 (可选)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist, Vector3
    from sensor_msgs.msg import JointState
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32MultiArray, Header
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[WARN] ROS2 未安装，将使用独立模式运行")


class Go2Simulator:
    """Go2 四足机器人 MuJoCo 仿真器"""
    
    # Go2关节命名（保持兼容）
    DOF_NAMES_GO2 = [
        "FL_hip_x", "FL_hip_y", "FL_knee",
        "FR_hip_x", "FR_hip_y", "FR_knee",
        "RL_hip_x", "RL_hip_y", "RL_knee",
        "RR_hip_x", "RR_hip_y", "RR_knee",
    ]

    # EVA02关节命名
    DOF_NAMES_EVA02 = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    # 默认使用Go2命名
    DOF_NAMES = DOF_NAMES_GO2

    def __init__(self, xml_path: Optional[str] = None, rl_model_path: Optional[str] = None,
                 action_scale: float = 0.25):
        if xml_path is None:
            xml_path = str(Path(__file__).resolve().parents[1] / "mjcf" / "go2" / "go2.xml")

        self.model = MjModel.from_xml_path(xml_path)
        self.data = MjData(self.model)

        # MuJoCo 线程安全锁 (mj_step 和 viewer.sync 不能同时执行)
        self.mj_lock = threading.Lock()

        # 自动检测关节命名（Go2 vs EVA02）
        joint_names = []
        for i in range(1, min(13, self.model.njnt)):  # 检查前12个关节
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                joint_names.append(name)

        # 判断是哪种命名
        if joint_names and 'joint' in joint_names[0]:
            self.DOF_NAMES = self.DOF_NAMES_EVA02
            print("[INFO] 检测到 EVA02 关节命名")
        elif joint_names and ('hip_x' in joint_names[0] or 'hip_y' in joint_names[0]):
            self.DOF_NAMES = self.DOF_NAMES_GO2
            print("[INFO] 检测到 Go2 关节命名")
        else:
            print(f"[WARN] 无法识别关节命名（第一个关节: {joint_names[0] if joint_names else 'None'}），使用Go2默认")
            self.DOF_NAMES = self.DOF_NAMES_GO2

        # 关节索引
        self._dof_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.DOF_NAMES
        ])

        # 执行器索引（尝试两种命名方式）
        actuator_ids = []
        for name in self.DOF_NAMES:
            # 尝试 name_motor 格式（EVA02）
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace('_joint', '_motor'))
                actuator_ids.append(actuator_id)
            except:
                # 尝试 name_act 格式（Go2）
                try:
                    actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_act")
                    actuator_ids.append(actuator_id)
                except:
                    print(f"[WARN] 找不到执行器: {name}")
                    actuator_ids.append(-1)

        self._actuator_ids = np.array(actuator_ids)

        # 根据机器人类型设置默认姿态
        if "eva02" in xml_path.lower():
            # EVA02 默认站立姿态 (从配置文件: thigh=0.9, calf=-1.6)
            self._default_dof_pos = np.array([
                0.0, 0.9, -1.6,   # FL (hip, thigh, calf)
                0.0, 0.9, -1.6,   # FR
                0.0, 0.9, -1.6,   # RL
                0.0, 0.9, -1.6,   # RR
            ])
        else:
            # Go2 默认站立姿态
            self._default_dof_pos = np.array([
                0.0, 1.0, -1.5,   # FL
                0.0, 1.0, -1.5,   # FR
                0.0, 1.0, -1.5,   # RL
                0.0, 1.0, -1.5,   # RR
            ])
        
        # RL 动作缩放 (关键参数!)
        self._action_scale = action_scale
        
        # 观测归一化缩放因子 (legged_gym/Unitree 标准)
        self._obs_scales = {
            "ang_vel": 0.25,              # 基座角速度缩放
            "dof_vel": 0.05,              # 关节速度缩放
            "commands": [2.0, 2.0, 0.25], # 命令缩放: vx*2, vy*2, vyaw*0.25
        }
        self._clip_obs = 100.0  # 观测裁剪上限
        
        # 指令速度 [vx, vy, vyaw]
        self._cmd_vel = np.zeros(3)
        self._cmd_lock = threading.Lock()
        
        # 步态相位变量
        self._gait_phase = 0.0
        self._gait_frequency = 2.0  # Hz
        self._gait_period = 1.0 / self._gait_frequency
        
        # 状态
        self._running = False
        self._dt = 0.005
        self._decimation = 4
        self._control_dt = self._dt * self._decimation
        
        # 足端接触历史
        self._foot_contacts = np.zeros(4)
        self._foot_site_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ["FL_foot_site", "FR_foot_site", "RL_foot_site", "RR_foot_site"]
        ])
        
        # ====== RL 模型 (可选) ======
        self._rl_model = None
        self._rl_model_path = rl_model_path
        self._last_action = np.zeros(12)  # RL 观测需要历史动作
        self._step_count = 0
        if rl_model_path and os.path.exists(rl_model_path):
            self.load_rl_model(rl_model_path)
        
        # 初始化到站立姿态
        self._init_stand()
    
    def _init_stand(self):
        """初始化站立姿态"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self._default_dof_pos
        mujoco.mj_forward(self.model, self.data)
        self._last_action = np.zeros(12)
    
    # ---- RL 模型 ----
    
    def load_rl_model(self, model_path: str):
        """
        加载训练好的 RL 模型 (ONNX 或 SB3 .zip)
        
        Args:
            model_path: 模型文件路径 (.zip / .onnx)
        """
        if model_path.endswith(".onnx"):
            self._load_onnx(model_path)
        elif model_path.endswith(".zip"):
            self._load_sb3(model_path)
        else:
            raise ValueError(f"不支持的模型格式: {model_path}，请使用 .zip (SB3) 或 .onnx")
        print(f"[RL] 模型已加载: {model_path}")
    
    def _load_onnx(self, path: str):
        """加载 ONNX 模型 (推荐，轻量无 Python 依赖)"""
        import onnxruntime as ort
        self._rl_model = ort.InferenceSession(path)
        self._rl_type = "onnx"
    
    def _load_sb3(self, path: str):
        """加载 Stable-Baselines3 模型"""
        from stable_baselines3 import PPO
        self._rl_model = PPO.load(path)
        self._rl_type = "sb3"
    
    @property
    def has_rl_model(self) -> bool:
        return self._rl_model is not None
    
    def _build_obs(self) -> np.ndarray:
        """
        构建 RL 观测向量 (45维) — legged_gym 归一化

        [0:3]   基座角速度 (×0.25)
        [3:6]   重力方向投影 (机体系)
        [6:9]   速度指令 [vx×2.0, vy×2.0, vyaw×0.25]
        [9:21]  关节位置 (相对默认姿态)
        [21:33] 关节速度 (×0.05)
        [33:45] 上一次动作 (裁剪至 [-clip_obs, clip_obs])
        """
        with self._cmd_lock:
            cmd = self._cmd_vel.copy()
        
        # 基座角速度 (归一化)
        base_ang_vel = self.data.qvel[3:6].copy() * self._obs_scales["ang_vel"]
        
        # 重力方向 (机体系)
        quat = self.data.qpos[3:7].copy()
        q_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]]) / np.sum(quat**2)
        gravity = np.array([0, 0, -1])
        projected_gravity = np.zeros(3)
        mujoco.mju_rotVecQuat(projected_gravity, gravity, q_inv)
        
        # 速度指令 (Isaac Lab: 原始物理值，无需额外缩放)
        cmd_scales = self._obs_scales["commands"]
        cmd_normalized = cmd * np.array(cmd_scales)
        
        # 关节状态 (相对默认姿态)
        joint_pos = (self.data.qpos[7:].copy() - self._default_dof_pos)
        joint_vel = self.data.qvel[6:].copy() * self._obs_scales["dof_vel"]
        
        # 上一次动作 (裁剪，防止异常值扩散)
        last_action = np.clip(self._last_action.copy(), -self._clip_obs, self._clip_obs)
        
        obs = np.concatenate([
            base_ang_vel,                   # 3 (归一化)
            projected_gravity,              # 3 (已归一化)
            cmd_normalized,                 # 3 (原始物理值)
            joint_pos,                      # 12 (相对值)
            joint_vel,                      # 12 (归一化)
            last_action,                    # 12 (裁剪后)
        ]).astype(np.float32)
        
        # 全局观测裁剪 (Isaac Lab 标准 clip_obs=100)
        obs = np.clip(obs, -self._clip_obs, self._clip_obs)
        
        return obs
    
    def _print_obs_diag(self, obs: np.ndarray):
        """诊断: 打印观测各分量范围"""
        labels = [
            ("ang_vel", 0, 3),
            ("gravity", 3, 6),
            ("cmd", 6, 9),
            ("joint_pos", 9, 21),
            ("joint_vel", 21, 33),
            ("last_action", 33, 45),
        ]
        parts = []
        for name, s, e in labels:
            v = obs[s:e]
            parts.append(f"{name}=[{v.min():.2f},{v.max():.2f}]")
        z = self.data.qpos[2]  # 基座高度
        parts.append(f"base_z={z:.3f}")
        print(f"[diag step {self._step_count:>4d}] " + " | ".join(parts))
    
    def _check_fallen(self) -> bool:
        """检测机器人是否摔倒 (基座高度过低或翻滚)"""
        z = self.data.qpos[2]
        # 重力方向在机体系 z 分量 (接近 -1 = 直立, 接近 0 = 侧翻)
        quat = self.data.qpos[3:7]
        q_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]]) / np.sum(quat**2)
        gravity_body = np.zeros(3)
        mujoco.mju_rotVecQuat(gravity_body, np.array([0, 0, -1]), q_inv)
        return z < 0.15 or abs(gravity_body[2]) < 0.3
    
    def _rl_inference(self, obs: np.ndarray) -> np.ndarray:
        """RL 模型推理 → 关节目标位置增量 (含 tanh 激活)"""
        if self._rl_type == "onnx":
            ort_inputs = {self._rl_model.get_inputs()[0].name: obs.reshape(1, -1)}
            action = self._rl_model.run(None, ort_inputs)[0][0]
        elif self._rl_type == "sb3":
            action, _ = self._rl_model.predict(obs, deterministic=True)
        else:
            action = np.zeros(12)
        
        # tanh 激活 (Isaac Lab / rsl_rl 导出模型不含 tanh，需手动添加)
        # action = np.tanh(action)  # 已禁用 - 训练时可能没用tanh
        
        # 动作 → 关节目标位置
        target_pos = self._default_dof_pos + action * self._action_scale
        target_pos = np.clip(
            target_pos,
            self.model.jnt_range[self._dof_ids, 0],
            self.model.jnt_range[self._dof_ids, 1],
        )
        
        # 每 500 步打印调试信息
        self._step_count += 1
        if self._step_count <= 5 or self._step_count % 500 == 0:
            print(f"[step {self._step_count:>4d}] "
                  f"action_range=[{action.min():.3f}, {action.max():.3f}] "
                  f"target_range=[{target_pos.min():.3f}, {target_pos.max():.3f}] "
                  f"obs_range=[{obs.min():.3f}, {obs.max():.3f}]")
        
        return target_pos
    
    def set_command(self, vx: float, vy: float, vyaw: float):
        """设置速度指令"""
        with self._cmd_lock:
            self._cmd_vel = np.array([vx, vy, vyaw])
    
    def get_state(self) -> dict:
        """获取当前机器人状态"""
        return {
            "base_pos": self.data.qpos[:3].copy(),
            "base_quat": self.data.qpos[3:7].copy(),
            "base_lin_vel": self.data.qvel[:3].copy(),
            "base_ang_vel": self.data.qvel[3:6].copy(),
            "joint_pos": self.data.qpos[7:].copy(),
            "joint_vel": self.data.qvel[6:].copy(),
            "foot_contacts": self._foot_contacts.copy(),
        }
    
    def step(self) -> dict:
        """执行一步仿真 + 控制
        
        ┌─ RL 模型可用? ──→ _rl_inference(_build_obs()) ──→ target_pos
        │
        └─ 步态引擎 ──────→ _compute_gait_targets(cmd) ──→ target_pos
        """
        # 获取当前指令
        with self._cmd_lock:
            cmd = self._cmd_vel.copy()
        
        # ====== RL 推理 / 步态引擎 (二选一) ======
        if self._rl_model is not None:
            obs = self._build_obs()
            target_pos = self._rl_inference(obs)
            self._last_action = (target_pos - self._default_dof_pos) / self._action_scale
            
            # 诊断: 每 200 步打印观测分项范围
            if self._step_count % 200 == 0:
                self._print_obs_diag(obs)
        else:
            self._gait_phase += self._control_dt * self._gait_frequency
            if self._gait_phase >= 1.0:
                self._gait_phase -= 1.0
            target_pos = self._compute_gait_targets(cmd)
        
        # 执行 PD 控制 (加锁保护，防止与 viewer.sync() 并发)
        with self.mj_lock:
            for i in range(self._decimation):
                self.data.ctrl[self._actuator_ids] = target_pos
                mujoco.mj_step(self.model, self.data)
        
        # 更新触地状态
        self._update_foot_contacts()
        
        # 摔倒检测 & 自动重置
        if self._check_fallen():
            print(f"\n[摔倒! step={self._step_count}] 自动重置站立姿态...")
            self._init_stand()
        
        return self.get_state()
    
    def _compute_gait_targets(self, cmd: np.ndarray) -> np.ndarray:
        """
        基于速度指令和步态相位计算关节目标位置
        
        实现对角小跑步态 (trot gait):
        - FL + RR 为一组，FR + RL 为另一组
        - 两组交替触地
        """
        vx, vy, vyaw = cmd
        target = self._default_dof_pos.copy()
        
        # 步态相位: 0~1
        phase = self._gait_phase
        
        # 对角小跑步态: FL/RR 和 FR/RL 交替
        # phase < 0.5: FL+RR 支撑, FR+RL 摆动
        # phase >= 0.5: FR+RL 支撑, FL+RR 摆动
        
        # 摆动高度
        swing_height = 0.08
        
        # 步幅 (与速度成正比)
        stride_x = vx * 0.05
        stride_y = vy * 0.03
        stride_yaw = vyaw * 0.02
        
        # 每条腿的相位偏移
        leg_phases = np.array([0.0, 0.5, 0.5, 0.0])  # FL, FR, RL, RR
        
        for leg_idx in range(4):
            leg_phase = (phase + leg_phases[leg_idx]) % 1.0
            
            # hip_x: 前后 (受 vx 影响)
            hip_x_offset = stride_x * math.sin(leg_phase * 2 * math.pi)
            
            # hip_y: 左右 (受 vy 和 vyaw 影响)
            if leg_idx in [0, 2]:  # 左腿
                hip_y_offset = stride_y * 0.5 + stride_yaw * 0.3
            else:  # 右腿
                hip_y_offset = -stride_y * 0.5 + stride_yaw * 0.3
            
            # knee: 弯曲 (摆动时收起，支撑时伸展)
            if leg_phase < 0.5:  # 支撑相
                knee_offset = 0.0
            else:  # 摆动相
                knee_offset = swing_height * math.sin(leg_phase * 2 * math.pi)
            
            # 应用偏移
            base_idx = leg_idx * 3
            target[base_idx + 0] += hip_x_offset     # hip_x
            target[base_idx + 1] += hip_y_offset     # hip_y
            target[base_idx + 2] += knee_offset * 1.5  # knee
        
        return np.clip(
            target,
            self.model.jnt_range[self._dof_ids, 0],
            self.model.jnt_range[self._dof_ids, 1],
        )
    
    def _update_foot_contacts(self):
        """更新足端接触状态"""
        self._foot_contacts = np.zeros(4)
        for i, site_id in enumerate(self._foot_site_ids):
            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                if contact.geom1 == site_id or contact.geom2 == site_id:
                    self._foot_contacts[i] = 1.0
                    break
    
    @property
    def running(self) -> bool:
        return self._running
    
    @running.setter
    def running(self, val: bool):
        self._running = val


class MujocoRosNode(Node if HAS_ROS2 else object):
    """
    ROS2 节点: 桥接手柄指令 → MuJoCo 仿真
    
    订阅:
      /cmd_vel (Twist) - 速度指令
    
    发布:
      /joint_states (JointState) - 关节状态
      /odom (Odometry) - 里程计
      /foot_contacts (Float32MultiArray) - 足端接触
    """
    
    def __init__(self, simulator: Go2Simulator):
        if HAS_ROS2:
            # ROS2 QoS 配置 (低延迟)
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            
            super().__init__("mujoco_ros_bridge")
            
            # 订阅 /cmd_vel
            self._cmd_sub = self.create_subscription(
                Twist, "/cmd_vel", self._cmd_callback, qos
            )
            
            # 发布者
            self._joint_pub = self.create_publisher(JointState, "/joint_states", qos)
            self._odom_pub = self.create_publisher(Odometry, "/odom", qos)
            self._contact_pub = self.create_publisher(
                Float32MultiArray, "/foot_contacts", qos
            )
            
            # 定时器: 50Hz 主循环
            self._timer = self.create_timer(1.0 / 50.0, self._timer_callback)
            
            self.get_logger().info("MuJoCo ROS Bridge 节点已启动")
        else:
            self._cmd_sub = None
            self._joint_pub = None
            self._odom_pub = None
            self._contact_pub = None
            self._timer = None
        
        self._sim = simulator
        self._step_count = 0
    
    def _cmd_callback(self, msg: Twist):
        """接收 /cmd_vel 指令"""
        vx = msg.linear.x
        vy = msg.linear.y
        vyaw = msg.angular.z
        self._sim.set_command(vx, vy, vyaw)
    
    def _timer_callback(self):
        """50Hz 主循环"""
        # 执行仿真步
        state = self._sim.step()
        self._step_count += 1
        
        if not HAS_ROS2:
            return
        
        # 发布关节状态
        self._publish_joint_state(state)
        
        # 发布里程计
        self._publish_odom(state)
        
        # 发布足端接触
        self._publish_foot_contacts(state)
    
    def _publish_joint_state(self, state: dict):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = Go2Simulator.DOF_NAMES
        msg.position = state["joint_pos"].tolist()
        msg.velocity = state["joint_vel"].tolist()
        self._joint_pub.publish(msg)
    
    def _publish_odom(self, state: dict):
        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        
        # 位置
        msg.pose.pose.position.x = float(state["base_pos"][0])
        msg.pose.pose.position.y = float(state["base_pos"][1])
        msg.pose.pose.position.z = float(state["base_pos"][2])
        
        # 姿态
        q = state["base_quat"]
        msg.pose.pose.orientation.w = float(q[0])
        msg.pose.pose.orientation.x = float(q[1])
        msg.pose.pose.orientation.y = float(q[2])
        msg.pose.pose.orientation.z = float(q[3])
        
        # 速度
        msg.twist.twist.linear.x = float(state["base_lin_vel"][0])
        msg.twist.twist.linear.y = float(state["base_lin_vel"][1])
        msg.twist.twist.linear.z = float(state["base_lin_vel"][2])
        msg.twist.twist.angular.x = float(state["base_ang_vel"][0])
        msg.twist.twist.angular.y = float(state["base_ang_vel"][1])
        msg.twist.twist.angular.z = float(state["base_ang_vel"][2])
        
        self._odom_pub.publish(msg)
    
    def _publish_foot_contacts(self, state: dict):
        msg = Float32MultiArray()
        msg.data = state["foot_contacts"].tolist()
        self._contact_pub.publish(msg)


# ====== 键盘输入 (pynput 全局捕获，不依赖 GLFW/终端) ======
_pynput_keys = set()
_pynput_listener = None

def _pynput_on_press(key):
    """pynput 按键回调"""
    try:
        from pynput.keyboard import Key, KeyCode
        # 方向键
        arrow_map = {Key.up: 'up', Key.down: 'down', Key.left: 'left', Key.right: 'right',
                     Key.space: 'space', Key.esc: 'esc'}
        name = arrow_map.get(key, None)
        if name:
            _pynput_keys.add(name)
            print(f"[按键] {name}", flush=True)
            return

        # WASD键 + QE键
        if isinstance(key, KeyCode):
            char = getattr(key, 'char', None)
            if char and char.lower() in ['w', 'a', 's', 'd', 'q', 'e']:
                _pynput_keys.add(char.lower())
                print(f"[按键] {char.lower()}", flush=True)
    except Exception:
        pass

def _pynput_on_release(key):
    try:
        from pynput.keyboard import Key, KeyCode
        arrow_map = {Key.up: 'up', Key.down: 'down', Key.left: 'left', Key.right: 'right',
                     Key.space: 'space', Key.esc: 'esc'}
        name = arrow_map.get(key, None)
        if name:
            _pynput_keys.discard(name)
            return

        # WASD键 + QE键
        if isinstance(key, KeyCode):
            char = getattr(key, 'char', None)
            if char and char.lower() in ['w', 'a', 's', 'd', 'q', 'e']:
                _pynput_keys.discard(char.lower())
    except Exception:
        pass

def _start_keys():
    global _pynput_listener
    try:
        from pynput.keyboard import Listener
        _pynput_listener = Listener(on_press=_pynput_on_press, on_release=_pynput_on_release)
        _pynput_listener.start()
        print("[键盘] pynput 全局监听已启动，方向键控制 (系统级，不需窗口焦点)")
    except ImportError:
        print("[键盘] pynput 未安装，运行: pip install pynput")
    except Exception as e:
        print(f"[键盘] 启动失败: {e}")

def _stop_keys():
    global _pynput_listener
    if _pynput_listener:
        _pynput_listener.stop()

def _get_keys() -> set:
    return _pynput_keys.copy()


def run_standalone(xml_path: Optional[str] = None, rl_model_path: Optional[str] = None,
                   action_scale: float = 0.25):
    """独立模式: 带渲染窗口，键盘控制 (使用 MuJoCo 原生 viewer)"""
    sim = Go2Simulator(xml_path, rl_model_path=rl_model_path, action_scale=action_scale)
    sim.running = True
    
    # 使用 MuJoCo 原生被动 viewer (与云深处相同)
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    
    # 启动全局键盘监听 (pynput)
    _start_keys()
    
    mode_str = "RL 模型" if sim.has_rl_model else "步态引擎"
    print("=" * 50)
    print(f"Go2 MuJoCo 仿真器 - 独立模式 ({mode_str})")
    print("键盘控制: ↑/↓前进后退 ←/→左转右转 Space停止 Esc退出")
    print("=" * 50)
    
    # 主循环
    last_time = time.time()
    step_count = 0
    
    try:
        while sim.running and viewer.is_running():
            # 获取按键 (pynput 全局监听)
            active_keys = _get_keys()
            if "esc" in active_keys:
                sim.running = False
                break
            if "esc" in active_keys:
                sim.running = False
                break
            
            # 处理键盘输入 (方向键)
            vx, vy, vyaw = 0.0, 0.0, 0.0
            if "up" in active_keys: vx += 1.0
            if "down" in active_keys: vx -= 1.0
            if "left" in active_keys: vyaw += 1.5
            if "right" in active_keys: vyaw -= 1.5
            if "space" in active_keys:
                vx = vy = vyaw = 0.0
            
            sim.set_command(vx, vy, vyaw)
            
            # 仿真步
            state = sim.step()
            step_count += 1
            
            # 渲染 (MuJoCo 原生 viewer: sync 而非 render)
            viewer.sync()
            
            # 控制频率
            current_time = time.time()
            elapsed = current_time - last_time
            target_dt = 1.0 / 50.0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            last_time = time.time()
            
            # 每 100 步打印状态
            if step_count % 100 == 0:
                pos = state["base_pos"]
                print(f"\r[vx={vx:.1f}, vy={vy:.1f}, vyaw={vyaw:.1f}] "
                      f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.3f})", end="")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        _stop_keys()
        viewer.close()
        print("\n仿真已停止")


def run_ros2(xml_path: Optional[str] = None, rl_model_path: Optional[str] = None,
              action_scale: float = 0.25):
    """ROS2 模式"""
    if not HAS_ROS2:
        print("[ERROR] ROS2 不可用，无法启动节点模式")
        sys.exit(1)
    
    rclpy.init()
    
    sim = Go2Simulator(xml_path, rl_model_path=rl_model_path, action_scale=action_scale)
    sim.running = True
    
    # 使用 MuJoCo 原生被动 viewer (与云深处相同)
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    
    node = MujocoRosNode(sim)
    
    # 在单独线程运行 ROS2 spin
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    print("=" * 50)
    print("Go2 MuJoCo + ROS2 桥接模式 (MuJoCo 原生 viewer)")
    print("等待 /cmd_vel 指令...")
    print("=" * 50)
    
    try:
        while sim.running and viewer.is_running():
            with sim.mj_lock:
                viewer.sync()
            time.sleep(0.002)  # ~500Hz 渲染
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        sim.running = False
        viewer.close()
        rclpy.shutdown()
        print("ROS2 节点已关闭")


def main():
    parser = argparse.ArgumentParser(description="四足机器人 MuJoCo + ROS2 桥接")
    parser.add_argument("--standalone", action="store_true",
                        help="独立模式 (无 ROS2，键盘控制)")
    parser.add_argument("--robot", type=str, default="go2",
                        help="机器人类型: go2, eva02 (默认: go2)")
    parser.add_argument("--xml", type=str, default=None,
                        help="MJCF 模型路径 (覆盖 --robot 选项)")
    parser.add_argument("--model", type=str, default=None,
                        help="RL 模型路径 (.zip / .onnx)，加载后替代步态引擎")
    parser.add_argument("--action-scale", type=float, default=0.5,
                        help="RL 动作缩放因子，默认 0.5，可尝试 0.25/1.0")
    args = parser.parse_args()

    # 如果没有指定xml，使用robot参数加载
    xml_path = args.xml
    if xml_path is None:
        try:
            from src.robot_config import RobotFactory
            config, xml_path = RobotFactory.create(args.robot)
            print(f"[INFO] 加载机器人: {config.name} ({config.description})")
        except Exception as e:
            print(f"[WARN] 无法加载机器人配置 '{args.robot}': {e}")
            print(f"[INFO] 使用默认路径...")
            xml_path = None

    if args.standalone or not HAS_ROS2:
        run_standalone(xml_path, args.model, args.action_scale)
    else:
        run_ros2(xml_path, args.model, args.action_scale)


if __name__ == "__main__":
    main()
