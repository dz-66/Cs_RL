#!/usr/bin/env python3
"""
对比训练配置和当前配置
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ros2_bridge.mujoco_ros_node import Go2Simulator
from src.robot_config import RobotFactory


config, xml_path = RobotFactory.create('eva02')
sim = Go2Simulator(xml_path, rl_model_path=None)

print("=" * 70)
print("📊 当前配置")
print("=" * 70)
print(f"\n默认姿态 (default_dof_pos):")
print(sim._default_dof_pos)
print(f"\n格式化显示:")
for i in range(4):
    leg = ['FL', 'FR', 'RL', 'RR'][i]
    idx = i * 3
    print(f"  {leg}: hip={sim._default_dof_pos[idx]:.2f}, "
          f"thigh={sim._default_dof_pos[idx+1]:.2f}, "
          f"calf={sim._default_dof_pos[idx+2]:.2f}")

print(f"\n观测缩放因子:")
print(f"  ang_vel_scale: {sim._obs_scales['ang_vel']}")
print(f"  dof_vel_scale: {sim._obs_scales['dof_vel']}")
print(f"  commands_scale: {sim._obs_scales['commands']}")
print(f"  clip_observations: {sim._clip_obs}")

print(f"\nAction scale: {sim._action_scale}")

print("\n" + "=" * 70)
print("❓ 请检查你的训练配置")
print("=" * 70)
print("""
请在你的训练代码中找到这些参数，并对比：

1. default_dof_pos（最重要！）
   在 legged_gym 中通常在：
   - cfg.init_state.default_joint_angles
   - 或 task config 中的 defaultJointAngles

2. 观测缩放
   - obs_scales.ang_vel
   - obs_scales.dof_vel
   - obs_scales.commands (lin_vel, ang_vel)

3. 动作缩放
   - control.action_scale

如果不匹配，机器人会一直"纠正"到训练时的姿态。

特别检查：
- 训练时是否使用了不同的站立高度？
- 关节顺序是否一致？(FL, FR, RL, RR)
- 关节命名是否一致？(hip/thigh/calf vs hip_x/hip_y/knee)
""")
