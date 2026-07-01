#!/usr/bin/env python3
"""
直接测试ONNX模型推理
模拟不同的输入，查看输出
"""
import numpy as np
import onnxruntime as ort

# 加载ONNX模型
model_path = "/home/tino66/Cs_RL/sim2sim/model/policy.onnx"
sess = ort.InferenceSession(model_path)

print("=" * 70)
print("🔍 ONNX模型推理测试")
print("=" * 70)

# 获取输入输出信息
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
input_shape = sess.get_inputs()[0].shape
output_shape = sess.get_outputs()[0].shape

print(f"\n模型信息:")
print(f"  输入: {input_name}, shape: {input_shape}")
print(f"  输出: {output_name}, shape: {output_shape}")

# 构建测试观测 (45维)
# [0:3]   基座角速度
# [3:6]   重力方向
# [6:9]   速度指令 ← 关键！
# [9:21]  关节位置
# [21:33] 关节速度
# [33:45] 上次动作

def build_test_obs(vx=0.0, vy=0.0, vyaw=0.0):
    """构建测试观测向量"""
    obs = np.zeros(45, dtype=np.float32)

    # 基座角速度 (假设静止)
    obs[0:3] = [0.0, 0.0, 0.0]

    # 重力方向 (直立)
    obs[3:6] = [0.0, 0.0, -1.0]

    # 速度指令 (关键部分)
    obs[6:9] = [vx, vy, vyaw]

    # 关节位置 (相对默认姿态，假设在默认位置)
    obs[9:21] = np.zeros(12)

    # 关节速度 (假设静止)
    obs[21:33] = np.zeros(12)

    # 上次动作 (假设为0)
    obs[33:45] = np.zeros(12)

    return obs


print("\n" + "=" * 70)
print("测试不同的速度指令")
print("=" * 70)

test_cases = [
    (0.0, 0.0, 0.0, "静止"),
    (1.0, 0.0, 0.0, "前进 vx=1.0"),
    (2.0, 0.0, 0.0, "快速前进 vx=2.0"),
    (0.0, 1.0, 0.0, "左移 vy=1.0"),
    (0.0, 0.0, 1.0, "左转 vyaw=1.0"),
    (1.0, 0.0, 0.5, "前进+左转"),
]

for vx, vy, vyaw, desc in test_cases:
    print(f"\n{'='*70}")
    print(f"📋 测试: {desc}")
    print(f"{'='*70}")

    # 构建观测
    obs = build_test_obs(vx, vy, vyaw)

    print(f"输入观测:")
    print(f"  速度指令 (obs[6:9]): [{obs[6]:.2f}, {obs[7]:.2f}, {obs[8]:.2f}]")
    print(f"  观测范围: [{obs.min():.3f}, {obs.max():.3f}]")

    # ONNX推理
    ort_inputs = {input_name: obs.reshape(1, -1)}
    action_raw = sess.run(None, ort_inputs)[0][0]

    # tanh激活
    action = np.tanh(action_raw)

    print(f"\n模型输出:")
    print(f"  原始输出范围: [{action_raw.min():.3f}, {action_raw.max():.3f}]")
    print(f"  tanh后范围: [{action.min():.3f}, {action.max():.3f}]")
    print(f"  平均幅度: {np.abs(action).mean():.3f}")
    print(f"  动作向量 (前4维): {action[:4]}")


print("\n" + "=" * 70)
print("💡 分析")
print("=" * 70)

# 对比静止和运动指令
obs_still = build_test_obs(0.0, 0.0, 0.0)
obs_move = build_test_obs(1.0, 0.0, 0.0)

action_still = np.tanh(sess.run(None, {input_name: obs_still.reshape(1, -1)})[0][0])
action_move = np.tanh(sess.run(None, {input_name: obs_move.reshape(1, -1)})[0][0])

diff = np.abs(action_move - action_still).mean()

print(f"\n对比分析:")
print(f"  静止指令的动作幅度: {np.abs(action_still).mean():.3f}")
print(f"  前进指令的动作幅度: {np.abs(action_move).mean():.3f}")
print(f"  两者差异: {diff:.3f}")

if diff < 0.05:
    print(f"\n❌ 问题确认：模型对速度指令不敏感！")
    print(f"   差异只有 {diff:.3f}，说明模型基本忽略了速度指令")
    print(f"\n可能原因:")
    print(f"  1. 模型训练时速度指令的权重太低")
    print(f"  2. 观测缩放因子与训练时不匹配，指令信号被稀释")
    print(f"  3. 模型训练未收敛，没有学会跟随指令")
elif diff < 0.2:
    print(f"\n⚠️  模型对速度指令有响应，但不强")
    print(f"   差异 {diff:.3f}，可能需要调整观测缩放因子")
else:
    print(f"\n✅ 模型正常响应速度指令")
    print(f"   差异 {diff:.3f}，说明模型能区分不同指令")

print("\n建议:")
print("  1. 检查训练配置中的 commands_scale")
print("  2. 尝试放大速度指令的缩放: commands_scale = [5.0, 5.0, 1.0]")
print("  3. 如果问题仍然存在，说明模型训练有问题，需要重新训练")
