#!/usr/bin/env python3
"""
诊断：检查零速度指令下的RL模型输出
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ros2_bridge.mujoco_ros_node import Go2Simulator
from src.robot_config import RobotFactory


def test_zero_command():
    """测试零速度指令下的行为"""

    print("=" * 70)
    print("🔍 零速度指令测试")
    print("=" * 70)

    config, xml_path = RobotFactory.create('eva02')
    print(f"✓ 加载机器人: {config.name}")

    model_path = str(Path(__file__).parent.parent / "model" / "policy.onnx")

    if not Path(model_path).exists():
        print(f"✗ 模型不存在: {model_path}")
        return

    print(f"✓ 加载模型: {model_path}")

    sim = Go2Simulator(xml_path, rl_model_path=model_path, action_scale=0.5)

    # 明确设置零速度指令
    sim.set_command(0.0, 0.0, 0.0)

    print("\n" + "=" * 70)
    print("🧪 测试：保持站立（零速度指令）")
    print("=" * 70)
    print("期望：机器人应该静止站立，动作输出应该很小\n")

    for i in range(20):
        # 构建观测
        obs = sim._build_obs()

        # 检查速度指令部分
        cmd_in_obs = obs[6:9]

        # RL推理
        if sim._rl_model:
            if sim._rl_type == "onnx":
                ort_inputs = {sim._rl_model.get_inputs()[0].name: obs.reshape(1, -1)}
                action_raw = sim._rl_model.run(None, ort_inputs)[0][0]

            action = np.tanh(action_raw)

            if i % 5 == 0:
                print(f"\n[Step {i:2d}]")
                print(f"  速度指令 (obs[6:9]): {cmd_in_obs}")
                print(f"  RL原始输出:   [{action_raw.min():6.3f}, {action_raw.max():6.3f}]")
                print(f"  tanh后动作:   [{action.min():6.3f}, {action.max():6.3f}]")
                print(f"  基座高度: {sim.data.qpos[2]:.4f} m")

        # 执行一步
        sim.step()

    print("\n" + "=" * 70)
    print("💡 分析")
    print("=" * 70)

    # 最后一次观测
    obs = sim._build_obs()
    cmd_in_obs = obs[6:9]

    print(f"\n1. 速度指令检查:")
    print(f"   观测中的指令: {cmd_in_obs}")

    if np.any(cmd_in_obs != 0):
        print(f"   ⚠️  警告：观测中的速度指令不为0！")
        print(f"   这可能是观测缩放的问题")
        print(f"   当前缩放: {sim._obs_scales['commands']}")
    else:
        print(f"   ✅ 速度指令正确为0")

    print(f"\n2. 可能的原因:")
    print(f"   a) 模型训练时没有学会'静止站立'")
    print(f"   b) 默认姿态不匹配，模型一直在纠正姿态")
    print(f"   c) 观测输入有偏差")

    print(f"\n3. 建议:")
    print(f"   - 检查训练时的 default_dof_pos 是否与当前一致")
    print(f"   - 尝试不同的 action_scale (当前: {sim._action_scale})")
    print(f"   - 查看训练日志，确认模型是否学会了静止站立")


if __name__ == "__main__":
    test_zero_command()
