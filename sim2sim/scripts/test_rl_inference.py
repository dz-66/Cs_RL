#!/usr/bin/env python3
"""
测试RL模型推理
检查模型输入输出是否正常
"""
import sys
import numpy as np
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ros2_bridge.mujoco_ros_node import Go2Simulator
from src.robot_config import RobotFactory


def test_rl_model(robot_name='eva02', model_path=None, steps=100):
    """测试RL模型推理"""

    print("=" * 70)
    print(f"🤖 RL模型推理测试 - {robot_name.upper()}")
    print("=" * 70)

    # 加载机器人
    config, xml_path = RobotFactory.create(robot_name)
    print(f"✓ 机器人: {config.name}")

    # 默认模型路径
    if model_path is None:
        model_path = str(Path(__file__).parent.parent / "model" / "policy.onnx")

    if not Path(model_path).exists():
        print(f"✗ 模型文件不存在: {model_path}")
        return

    print(f"✓ 模型: {model_path}")

    # 创建仿真器（加载RL模型）
    print(f"\n加载仿真器...")
    sim = Go2Simulator(xml_path, rl_model_path=model_path, action_scale=0.25)

    if not sim.has_rl_model:
        print("✗ RL模型加载失败")
        return

    print(f"✓ RL模型已加载")
    print(f"  action_scale: {sim._action_scale}")

    # 设置测试指令（静止站立）
    sim.set_command(0.0, 0.0, 0.0)

    print(f"\n" + "=" * 70)
    print(f"🧪 运行 {steps} 步推理（静止站立）")
    print("=" * 70)

    base_heights = []
    action_ranges = []
    obs_ranges = []

    for i in range(steps):
        state = sim.step()

        # 记录数据
        base_z = state['base_pos'][2]
        base_heights.append(base_z)

        if i % 20 == 0:
            # 手动构建观测查看
            obs = sim._build_obs()
            action_raw = (sim.data.ctrl[sim._actuator_ids] - sim._default_dof_pos) / sim._action_scale

            print(f"\n[Step {i:3d}]")
            print(f"  基座高度: {base_z:.4f} m")
            print(f"  观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
            print(f"  动作范围: [{action_raw.min():.3f}, {action_raw.max():.3f}]")
            print(f"  目标位置: [{sim.data.ctrl[sim._actuator_ids].min():.3f}, "
                  f"{sim.data.ctrl[sim._actuator_ids].max():.3f}]")

            # 关节位置偏差
            joint_pos = state['joint_pos']
            joint_error = joint_pos - sim._default_dof_pos
            print(f"  关节偏差: [{joint_error.min():.3f}, {joint_error.max():.3f}] "
                  f"mean={joint_error.mean():.3f}")

            action_ranges.append((action_raw.min(), action_raw.max()))
            obs_ranges.append((obs.min(), obs.max()))

    # 分析结果
    base_heights = np.array(base_heights)

    print("\n" + "=" * 70)
    print("📊 分析结果")
    print("=" * 70)

    print(f"\n基座高度:")
    print(f"  初始: {base_heights[0]:.4f} m")
    print(f"  最终: {base_heights[-1]:.4f} m")
    print(f"  最小: {base_heights.min():.4f} m")
    print(f"  最大: {base_heights.max():.4f} m")
    print(f"  平均: {base_heights.mean():.4f} m")
    print(f"  标准差: {base_heights.std():.4f} m")

    # 判断稳定性
    print(f"\n稳定性评估:")

    height_drop = base_heights[0] - base_heights[-1]
    if height_drop > 0.1:
        print(f"  ❌ 高度下降 {height_drop:.3f} m - 机器人正在下蹲/摔倒")
    elif base_heights.min() < 0.20:
        print(f"  ❌ 最低高度 {base_heights.min():.3f} m - 机器人摔倒")
    elif base_heights.std() > 0.05:
        print(f"  ⚠️  高度波动较大 (std={base_heights.std():.3f}) - 不稳定")
    else:
        print(f"  ✅ 机器人保持稳定站立")

    # 检查动作是否合理
    print(f"\n动作输出:")
    if action_ranges:
        all_mins = [r[0] for r in action_ranges]
        all_maxs = [r[1] for r in action_ranges]
        print(f"  范围: [{min(all_mins):.3f}, {max(all_maxs):.3f}]")

        if max(all_maxs) < 0.01 and min(all_mins) > -0.01:
            print(f"  ⚠️  动作几乎为零 - 模型可能未激活")
        elif max(all_maxs) > 5.0 or min(all_mins) < -5.0:
            print(f"  ⚠️  动作过大 - 可能缺少tanh激活或action_scale不匹配")

    # 常见问题诊断
    print("\n" + "=" * 70)
    print("🔍 常见问题诊断")
    print("=" * 70)

    issues_found = False

    # 1. 检查默认姿态是否匹配
    print("\n1. 默认姿态检查:")
    print(f"   当前: {sim._default_dof_pos}")
    print(f"   训练时是否使用相同的default_dof_pos？")

    # 2. 检查action_scale
    print(f"\n2. Action Scale检查:")
    print(f"   当前: {sim._action_scale}")
    print(f"   训练时的action_scale是多少？")
    print(f"   建议: 尝试 0.25 / 0.5 / 1.0")

    # 3. 检查观测缩放
    print(f"\n3. 观测缩放检查:")
    print(f"   ang_vel: {sim._obs_scales['ang_vel']}")
    print(f"   dof_vel: {sim._obs_scales['dof_vel']}")
    print(f"   commands: {sim._obs_scales['commands']}")
    print(f"   训练时是否使用相同的缩放？")

    # 4. ONNX模型激活函数
    print(f"\n4. 模型激活函数检查:")
    print(f"   你的ONNX模型是否包含tanh激活？")
    print(f"   如果训练时使用tanh但导出时未包含，需要在推理后手动添加")
    print(f"   位置: mujoco_ros_node.py:317 (已注释)")

    # 5. 关节命名映射
    print(f"\n5. 关节命名映射:")
    print(f"   检测到: {sim.DOF_NAMES[:3]}...")
    print(f"   训练时的关节顺序是否一致？")

    print("\n" + "=" * 70)
    print("💡 调试建议")
    print("=" * 70)
    print("""
1. 对比训练配置文件（legged_gym的cfg文件）
   - 确认 default_dof_pos 一致
   - 确认 action_scale 一致
   - 确认 obs_scales 一致

2. 检查ONNX导出时是否包含策略头的激活函数
   - 如果遗漏，取消注释 mujoco_ros_node.py:317 的 tanh

3. 尝试不同的action_scale:
   python -m src.ros2_bridge.mujoco_ros_node --standalone --robot eva02 \\
       --model model/policy.onnx --action-scale 0.5

4. 查看训练日志中的reward曲线
   - 如果训练未收敛，模型本身可能有问题

5. 验证ONNX模型输入输出形状:
   python -c "import onnxruntime as ort; \\
       sess = ort.InferenceSession('model/policy.onnx'); \\
       print('Input:', sess.get_inputs()[0].shape); \\
       print('Output:', sess.get_outputs()[0].shape)"
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="eva02")
    parser.add_argument("--model", default=None)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    test_rl_model(args.robot, args.model, args.steps)
