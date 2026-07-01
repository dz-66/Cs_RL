#!/usr/bin/env python3
"""
测试不同的 action_scale，找出最稳定的值
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ros2_bridge.mujoco_ros_node import Go2Simulator
from src.robot_config import RobotFactory

config, xml_path = RobotFactory.create('eva02')
model_path = "/home/tino66/Cs_RL/sim2sim/model/policy.onnx"

print("=" * 80)
print("🔍 测试不同的 action_scale")
print("=" * 80)

test_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

results = []

for scale in test_scales:
    print(f"\n{'='*80}")
    print(f"测试 action_scale = {scale}")
    print(f"{'='*80}")

    sim = Go2Simulator(xml_path, rl_model_path=model_path, action_scale=scale)

    # 测试前进指令
    sim.set_command(1.0, 0.0, 0.0)

    base_heights = []
    velocities = []

    for i in range(100):
        state = sim.step()
        base_heights.append(state['base_pos'][2])

        # 计算基座速度
        if i > 0:
            vel = abs(state['base_pos'][0] - prev_x) / sim._control_dt
            velocities.append(vel)
        prev_x = state['base_pos'][0]

        # 检测摔倒或乱飞
        if base_heights[-1] < 0.15 or base_heights[-1] > 0.6:
            print(f"  ❌ step {i}: 基座高度异常 {base_heights[-1]:.3f}m")
            break

    base_heights = np.array(base_heights)
    velocities = np.array(velocities) if velocities else np.array([0])

    # 评估稳定性
    height_stable = base_heights.std() < 0.05
    height_ok = base_heights.min() > 0.20 and base_heights.max() < 0.50
    has_motion = velocities.mean() > 0.01

    score = 0
    if height_stable: score += 30
    if height_ok: score += 40
    if has_motion: score += 30

    print(f"\n  结果:")
    print(f"    高度: {base_heights.mean():.3f}±{base_heights.std():.3f}m")
    print(f"    平均速度: {velocities.mean():.3f} m/s")
    print(f"    {'✅' if height_stable else '❌'} 高度稳定")
    print(f"    {'✅' if height_ok else '❌'} 高度正常")
    print(f"    {'✅' if has_motion else '❌'} 有运动响应")
    print(f"    得分: {score}/100")

    results.append({
        'scale': scale,
        'height_mean': base_heights.mean(),
        'height_std': base_heights.std(),
        'velocity': velocities.mean(),
        'stable': height_stable and height_ok,
        'score': score,
    })

# 排序
results.sort(key=lambda x: x['score'], reverse=True)

print("\n" + "=" * 80)
print("📊 测试结果汇总")
print("=" * 80)

print(f"\n{'scale':<8} {'高度':<10} {'稳定性':<10} {'速度':<10} {'得分':<8} {'状态':<6}")
print("-" * 80)

for r in results:
    status = "✅" if r['stable'] else "❌"
    print(f"{r['scale']:<8.1f} {r['height_mean']:>6.3f}m {r['height_std']:>8.3f} "
          f"{r['velocity']:>8.3f} {r['score']:>6}/100 {status:>8}")

print("\n" + "=" * 80)
print("💡 推荐配置")
print("=" * 80)

best = results[0]
print(f"\n最佳 action_scale: {best['scale']}")
print(f"得分: {best['score']}/100")

if best['score'] < 70:
    print(f"\n⚠️  即使最佳配置得分也不高")
    print(f"\n可能的问题:")
    print(f"  1. 模型训练质量不高")
    print(f"  2. PD增益需要调整")
    print(f"  3. 观测输入仍有偏差")
    print(f"\n建议:")
    print(f"  - 降低 PD 的 kp: 100 → 50")
    print(f"  - 增加 kd: 5 → 8")
    print(f"  - 检查训练日志，确认模型收敛")
