#!/usr/bin/env python3
"""
测试不同的观测缩放因子
找出让模型输出最小动作的配置
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ros2_bridge.mujoco_ros_node import Go2Simulator


# 常见的观测缩放配置
TEST_SCALES = {
    "当前配置": {
        "ang_vel": 0.5,
        "dof_vel": 0.2,
        "commands": [2.0, 2.0, 0.25],
    },
    "IsaacGym标准": {
        "ang_vel": 0.25,
        "dof_vel": 0.05,
        "commands": [2.0, 2.0, 0.25],
    },
    "较小缩放": {
        "ang_vel": 0.1,
        "dof_vel": 0.02,
        "commands": [1.0, 1.0, 0.1],
    },
    "较大缩放": {
        "ang_vel": 1.0,
        "dof_vel": 0.5,
        "commands": [3.0, 3.0, 0.5],
    },
    "无缩放": {
        "ang_vel": 1.0,
        "dof_vel": 1.0,
        "commands": [1.0, 1.0, 1.0],
    },
}


def test_scales(name, scales, xml_path, model_path, steps=30):
    """测试一组观测缩放"""

    print(f"\n{'='*70}")
    print(f"🧪 测试: {name}")
    print(f"{'='*70}")
    print(f"ang_vel: {scales['ang_vel']}, dof_vel: {scales['dof_vel']}, commands: {scales['commands']}")

    sim = Go2Simulator(xml_path, rl_model_path=model_path, action_scale=0.5)

    # 修改观测缩放
    sim._obs_scales = scales

    # 零速度指令
    sim.set_command(0.0, 0.0, 0.0)

    action_magnitudes = []

    for i in range(steps):
        obs = sim._build_obs()

        # RL推理
        if sim._rl_type == "onnx":
            ort_inputs = {sim._rl_model.get_inputs()[0].name: obs.reshape(1, -1)}
            action_raw = sim._rl_model.run(None, ort_inputs)[0][0]

        action = np.tanh(action_raw)
        action_magnitudes.append(np.abs(action).mean())

        sim.step()

    action_magnitudes = np.array(action_magnitudes)

    # 评分：动作越小越好
    avg_action = action_magnitudes.mean()
    score = max(0, 100 - avg_action * 100)

    print(f"\n📊 结果:")
    print(f"  平均动作幅度: {avg_action:.3f}")
    print(f"  动作范围: [{action_magnitudes.min():.3f}, {action_magnitudes.max():.3f}]")
    print(f"  基座高度: {sim.data.qpos[2]:.3f} m")
    print(f"  得分: {score:.0f}/100 {'✅' if avg_action < 0.3 else '❌'}")

    return {
        'name': name,
        'scales': scales,
        'action_mean': avg_action,
        'action_min': action_magnitudes.min(),
        'action_max': action_magnitudes.max(),
        'score': score,
    }


def main():
    xml_path = "/home/tino66/Cs_RL/sim2sim/src/mjcf/eva02/eva02_detailed.xml"
    model_path = "/home/tino66/Cs_RL/sim2sim/model/policy.onnx"

    if not Path(model_path).exists():
        print(f"错误: 模型不存在 {model_path}")
        return

    print("=" * 70)
    print("🔍 测试不同的观测缩放因子")
    print("=" * 70)
    print("目标: 找到让模型在零速度下输出最小动作的配置\n")

    results = []

    for name, scales in TEST_SCALES.items():
        try:
            result = test_scales(name, scales, xml_path, model_path)
            results.append(result)
        except Exception as e:
            print(f"✗ 测试失败: {e}")

    # 排序
    results.sort(key=lambda x: x['action_mean'])

    # 汇总
    print("\n" + "=" * 70)
    print("📊 测试结果汇总 (按动作幅度排序)")
    print("=" * 70)

    print(f"\n{'配置':<20} {'平均动作':>10} {'范围':>20} {'得分':>8}")
    print("-" * 70)

    for r in results:
        status = "✅" if r['action_mean'] < 0.3 else "❌"
        print(f"{r['name']:<20} {r['action_mean']:>10.3f} "
              f"[{r['action_min']:>5.3f}, {r['action_max']:>5.3f}] "
              f"{r['score']:>7.0f} {status}")

    # 推荐
    print("\n" + "=" * 70)
    print("💡 推荐配置")
    print("=" * 70)

    best = results[0]
    print(f"\n最佳配置: {best['name']}")
    print(f"平均动作: {best['action_mean']:.3f}")
    print(f"得分: {best['score']:.0f}/100")

    if best['action_mean'] > 0.5:
        print(f"\n⚠️  警告: 即使最佳配置的动作仍然很大 ({best['action_mean']:.2f})")
        print(f"\n这说明问题不在观测缩放，可能是:")
        print(f"  1. 模型训练时没有学会'静止站立'任务")
        print(f"  2. 训练环境与sim2sim差异太大")
        print(f"  3. 模型期望持续收到非零速度指令")
        print(f"  4. 需要查看训练配置和奖励函数设置")
    else:
        print(f"\n✅ 找到合适的配置！")
        print(f"\n修改代码:")
        print(f"  文件: sim2sim/src/ros2_bridge/mujoco_ros_node.py:149-153")
        print(f"  修改为:")
        print(f"    self._obs_scales = {{")
        print(f"        'ang_vel': {best['scales']['ang_vel']},")
        print(f"        'dof_vel': {best['scales']['dof_vel']},")
        print(f"        'commands': {best['scales']['commands']},")
        print(f"    }}")


if __name__ == "__main__":
    main()
