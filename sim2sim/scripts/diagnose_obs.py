#!/usr/bin/env python3
"""
RL观测诊断工具
检查观测向量是否与训练时匹配
"""
import sys
import numpy as np
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ros2_bridge.mujoco_ros_node import Go2Simulator
from src.robot_config import RobotFactory


def diagnose_observation(robot_name='eva02', steps=50):
    """诊断观测向量构建"""

    print("=" * 60)
    print(f"🔍 RL观测诊断 - {robot_name.upper()}")
    print("=" * 60)

    # 加载机器人
    try:
        config, xml_path = RobotFactory.create(robot_name)
        print(f"✓ 加载机器人: {config.name}")
        print(f"  MJCF: {xml_path}")
        print(f"  默认姿态: {config.default_pose}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return

    # 创建仿真器（不加载RL模型，只检查观测）
    sim = Go2Simulator(xml_path, rl_model_path=None)

    print(f"\n📊 仿真器配置:")
    print(f"  关节数: {len(sim.DOF_NAMES)}")
    print(f"  关节名: {sim.DOF_NAMES[:3]}... (共{len(sim.DOF_NAMES)}个)")
    print(f"  观测缩放: ang_vel={sim._obs_scales['ang_vel']}, "
          f"dof_vel={sim._obs_scales['dof_vel']}, "
          f"cmd={sim._obs_scales['commands']}")

    # 设置测试指令
    sim.set_command(1.0, 0.0, 0.0)  # 前进指令

    print(f"\n🧪 运行 {steps} 步仿真...")
    print("-" * 60)

    obs_history = []
    for i in range(steps):
        # 手动触发一次观测构建（不执行RL推理）
        obs = sim._build_obs()
        obs_history.append(obs)

        # 使用步态引擎控制（保持站立）
        sim.set_command(0.0, 0.0, 0.0)  # 静止
        state = sim.step()

        if i % 10 == 0:
            print(f"\n[Step {i:3d}]")
            print(f"  基座高度: {state['base_pos'][2]:.4f} m")
            print(f"  观测向量形状: {obs.shape}")
            print(f"  观测范围: [{obs.min():.3f}, {obs.max():.3f}]")

            # 分段检查
            labels = [
                ("基座角速度", 0, 3),
                ("重力方向", 3, 6),
                ("速度指令", 6, 9),
                ("关节位置", 9, 21),
                ("关节速度", 21, 33),
                ("上次动作", 33, 45),
            ]

            for name, start, end in labels:
                segment = obs[start:end]
                print(f"  {name:12s}: [{segment.min():6.3f}, {segment.max():6.3f}] "
                      f"mean={segment.mean():6.3f} std={segment.std():6.3f}")

    # 统计分析
    obs_array = np.array(obs_history)

    print("\n" + "=" * 60)
    print("📈 统计分析（全过程）")
    print("=" * 60)

    labels = [
        ("基座角速度", 0, 3),
        ("重力方向", 3, 6),
        ("速度指令", 6, 9),
        ("关节位置", 9, 21),
        ("关节速度", 21, 33),
        ("上次动作", 33, 45),
    ]

    for name, start, end in labels:
        segment = obs_array[:, start:end]
        print(f"\n{name}:")
        print(f"  范围: [{segment.min():.4f}, {segment.max():.4f}]")
        print(f"  均值: {segment.mean():.4f} ± {segment.std():.4f}")
        if segment.std() < 1e-6:
            print(f"  ⚠️  几乎无变化 - 可能有问题")

    # 异常检测
    print("\n" + "=" * 60)
    print("⚠️  潜在问题检测")
    print("=" * 60)

    issues = []

    # 1. 检查重力方向
    gravity_seg = obs_array[:, 3:6]
    if abs(gravity_seg[:, 2].mean() - (-1.0)) > 0.1:
        issues.append(f"重力方向异常: z轴应接近-1.0，实际={gravity_seg[:, 2].mean():.3f}")

    # 2. 检查关节位置是否合理
    joint_pos = obs_array[:, 9:21]
    if np.abs(joint_pos).max() > 10:
        issues.append(f"关节位置过大: max={joint_pos.max():.3f} (可能未减去默认姿态)")

    # 3. 检查观测是否全为零
    for name, start, end in labels:
        segment = obs_array[:, start:end]
        if np.abs(segment).max() < 1e-6:
            issues.append(f"{name} 全为零")

    # 4. 检查NaN/Inf
    if np.any(np.isnan(obs_array)) or np.any(np.isinf(obs_array)):
        issues.append("存在 NaN 或 Inf 值")

    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ 未发现明显异常")

    # 训练配置建议
    print("\n" + "=" * 60)
    print("💡 训练配置对照")
    print("=" * 60)
    print("请确认你的训练环境使用了相同的配置：")
    print(f"""
class Obs:
    ang_vel_scale = {sim._obs_scales['ang_vel']}
    dof_vel_scale = {sim._obs_scales['dof_vel']}
    commands_scale = {sim._obs_scales['commands']}
    clip_observations = {sim._clip_obs}

class DefaultPose:
    dof_pos = {sim._default_dof_pos.tolist()}

class Control:
    action_scale = {sim._action_scale}
""")

    print("\n如果训练时使用了不同的缩放因子，请修改 mujoco_ros_node.py")
    print("例如：self._obs_scales = {'ang_vel': 0.25, 'dof_vel': 0.05, ...}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="eva02", help="机器人名称")
    parser.add_argument("--steps", type=int, default=50, help="仿真步数")
    args = parser.parse_args()

    diagnose_observation(args.robot, args.steps)
