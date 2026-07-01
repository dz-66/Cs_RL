#!/usr/bin/env python3
"""
精细调优：找到让静止动作最小的观测配置
通过网格搜索找到最佳的观测缩放组合
"""
import numpy as np
import onnxruntime as ort
from itertools import product

model_path = "/home/tino66/Cs_RL/sim2sim/model/policy.onnx"
sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name

def build_obs(vx=0.0, vy=0.0, vyaw=0.0, ang_vel_scale=0.1, dof_vel_scale=0.02):
    """构建观测向量"""
    obs = np.zeros(45, dtype=np.float32)
    obs[0:3] = np.array([0.0, 0.0, 0.0]) * ang_vel_scale  # 基座角速度
    obs[3:6] = [0.0, 0.0, -1.0]                           # 重力方向
    obs[6:9] = [vx, vy, vyaw]                             # 速度指令
    obs[9:21] = np.zeros(12)                              # 关节位置（相对默认）
    obs[21:33] = np.zeros(12) * dof_vel_scale            # 关节速度
    obs[33:45] = np.zeros(12)                             # 上次动作
    return obs

print("=" * 80)
print("🔍 精细调优：减少静止时的基础动作")
print("=" * 80)

# 网格搜索参数
ang_vel_scales = [0.05, 0.1, 0.25, 0.5, 1.0]
dof_vel_scales = [0.01, 0.02, 0.05, 0.1, 0.2]

print("\n正在测试 {} 种配置组合...".format(len(ang_vel_scales) * len(dof_vel_scales)))

results = []

for ang_scale, dof_scale in product(ang_vel_scales, dof_vel_scales):
    # 静止观测
    obs_still = build_obs(0.0, 0.0, 0.0, ang_scale, dof_scale)
    action_still_raw = sess.run(None, {input_name: obs_still.reshape(1, -1)})[0][0]
    action_still = np.tanh(action_still_raw)
    still_magnitude = np.abs(action_still).mean()

    # 前进观测
    obs_move = build_obs(1.0, 0.0, 0.0, ang_scale, dof_scale)
    action_move_raw = sess.run(None, {input_name: obs_move.reshape(1, -1)})[0][0]
    action_move = np.tanh(action_move_raw)
    move_magnitude = np.abs(action_move).mean()

    # 响应度 = 前进动作 - 静止动作
    responsiveness = move_magnitude - still_magnitude

    # 评分：静止动作越小越好，但响应度也要足够
    score = 100 - still_magnitude * 100 + responsiveness * 50

    results.append({
        'ang_vel_scale': ang_scale,
        'dof_vel_scale': dof_scale,
        'still_magnitude': still_magnitude,
        'move_magnitude': move_magnitude,
        'responsiveness': responsiveness,
        'score': score,
    })

# 排序：优先静止动作小，其次响应度大
results.sort(key=lambda x: (x['still_magnitude'], -x['responsiveness']))

print("\n" + "=" * 80)
print("📊 Top 10 配置（按静止动作幅度排序）")
print("=" * 80)
print(f"\n{'排名':<4} {'ang_vel':<8} {'dof_vel':<8} {'静止动作':<10} {'前进动作':<10} {'响应度':<10}")
print("-" * 80)

for i, r in enumerate(results[:10], 1):
    print(f"{i:<4} {r['ang_vel_scale']:<8.2f} {r['dof_vel_scale']:<8.2f} "
          f"{r['still_magnitude']:<10.3f} {r['move_magnitude']:<10.3f} "
          f"{r['responsiveness']:<10.3f}")

# 推荐配置
best = results[0]

print("\n" + "=" * 80)
print("💡 推荐配置")
print("=" * 80)

print(f"\n最佳配置:")
print(f"  ang_vel_scale: {best['ang_vel_scale']}")
print(f"  dof_vel_scale: {best['dof_vel_scale']}")
print(f"  静止时动作幅度: {best['still_magnitude']:.3f}")
print(f"  前进时动作幅度: {best['move_magnitude']:.3f}")
print(f"  响应度: {best['responsiveness']:.3f}")

if best['still_magnitude'] > 0.2:
    print(f"\n⚠️  即使最佳配置，静止动作仍然较大 ({best['still_magnitude']:.3f})")
    print(f"\n这说明问题的根源可能是：")
    print(f"  1. 模型训练时的默认姿态与当前不同")
    print(f"  2. 模型没有被训练成'零速度=保持姿态'")
    print(f"  3. 观测中的其他分量（如重力方向、关节位置）有偏差")
    print(f"\n建议：")
    print(f"  - 检查 default_dof_pos 是否与训练时完全一致")
    print(f"  - 尝试在观测的 joint_pos (obs[9:21]) 中添加小的随机偏移")
    print(f"  - 考虑重新训练模型，强化'静止站立'奖励")
else:
    print(f"\n✅ 找到合适的配置！")

print(f"\n修改代码:")
print(f"  文件: sim2sim/src/ros2_bridge/mujoco_ros_node.py:149-153")
print(f"  修改为:")
print(f"    self._obs_scales = {{")
print(f"        'ang_vel': {best['ang_vel_scale']},")
print(f"        'dof_vel': {best['dof_vel_scale']},")
print(f"        'commands': [1.0, 1.0, 0.1],  # 保持不变")
print(f"    }}")

# 详细分析最佳配置
print("\n" + "=" * 80)
print("🔬 最佳配置详细分析")
print("=" * 80)

obs_still = build_obs(0.0, 0.0, 0.0, best['ang_vel_scale'], best['dof_vel_scale'])
action_still_raw = sess.run(None, {input_name: obs_still.reshape(1, -1)})[0][0]
action_still = np.tanh(action_still_raw)

joint_names = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

print(f"\n静止指令下的12维动作输出:")
for i, name in enumerate(joint_names):
    print(f"  {name:12s}: {action_still[i]:7.3f}")

print(f"\n如果这些值仍然很大，说明模型认为当前姿态不是'正确'的静止姿态")
