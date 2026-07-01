#!/usr/bin/env python3
"""
完整诊断：从键盘输入到RL输出的全链路
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ros2_bridge.mujoco_ros_node import Go2Simulator
from src.robot_config import RobotFactory

config, xml_path = RobotFactory.create('eva02')
model_path = "/home/tino66/Cs_RL/sim2sim/model/policy.onnx"

sim = Go2Simulator(xml_path, rl_model_path=model_path, action_scale=0.5)

print("=" * 80)
print("🔍 完整链路诊断：键盘输入 → RL输出")
print("=" * 80)

# 模拟键盘输入的不同速度
test_commands = [
    (0.0, 0.0, 0.0, "静止"),
    (1.0, 0.0, 0.0, "前进 vx=1.0"),
    (2.0, 0.0, 0.0, "快速前进 vx=2.0"),
    (-1.0, 0.0, 0.0, "后退 vx=-1.0"),
    (0.0, 0.0, 1.0, "左转 vyaw=1.0"),
]

for vx, vy, vyaw, desc in test_commands:
    print(f"\n{'='*80}")
    print(f"测试: {desc}")
    print(f"{'='*80}")

    # 1. 设置指令
    sim.set_command(vx, vy, vyaw)
    print(f"\n1. set_command({vx}, {vy}, {vyaw})")
    print(f"   sim._cmd_vel = {sim._cmd_vel}")

    # 2. 构建观测
    obs = sim._build_obs()
    print(f"\n2. 构建观测 _build_obs()")
    print(f"   obs[6:9] (速度指令部分) = {obs[6:9]}")
    print(f"   期望值 = {np.array([vx, vy, vyaw]) * np.array(sim._obs_scales['commands'])}")

    # 3. RL推理
    if sim._rl_type == "onnx":
        ort_inputs = {sim._rl_model.get_inputs()[0].name: obs.reshape(1, -1)}
        action_raw = sim._rl_model.run(None, ort_inputs)[0][0]

    action = np.tanh(action_raw)

    print(f"\n3. RL模型推理")
    print(f"   原始输出范围: [{action_raw.min():.3f}, {action_raw.max():.3f}]")
    print(f"   tanh后范围: [{action.min():.3f}, {action.max():.3f}]")
    print(f"   平均幅度: {np.abs(action).mean():.3f}")

    # 4. 计算目标位置
    target_pos = sim._default_dof_pos + action * sim._action_scale

    print(f"\n4. 计算关节目标位置")
    print(f"   默认姿态: {sim._default_dof_pos[:3]} (前3维)")
    print(f"   动作增量: {action[:3] * sim._action_scale} (前3维)")
    print(f"   目标位置: {target_pos[:3]} (前3维)")

    # 对比静止和运动的差异
    if vx == 0.0 and vy == 0.0 and vyaw == 0.0:
        action_still = action.copy()
    else:
        diff = np.abs(action - action_still).mean()
        print(f"\n5. 与静止状态对比")
        print(f"   动作差异: {diff:.3f}")
        if diff < 0.1:
            print(f"   ⚠️  差异很小！指令信号可能太弱")
        else:
            print(f"   ✅ 有明显差异")

print("\n" + "=" * 80)
print("💡 诊断结论")
print("=" * 80)

print(f"""
当前配置:
  - commands_scale: {sim._obs_scales['commands']}
  - action_scale: {sim._action_scale}
  - ang_vel_scale: {sim._obs_scales['ang_vel']}
  - dof_vel_scale: {sim._obs_scales['dof_vel']}

如果上面显示:
1. ✅ sim._cmd_vel 正确更新
2. ✅ obs[6:9] 与期望值匹配
3. ✅ RL输出在不同指令下有差异
4. ❌ 但实际运行时没有响应

那么可能是:
  - PD控制器的增益太高，过度阻尼了运动
  - action_scale 太小 (当前={sim._action_scale})
  - 关节目标位置被裁剪到了限位

建议:
  1. 尝试更大的 action_scale: --action-scale 1.0
  2. 检查 MJCF 文件中的 PD 增益 (kp, kv)
  3. 检查关节限位范围
""")
