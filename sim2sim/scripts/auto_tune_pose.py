#!/usr/bin/env python3
"""
自动测试不同的默认姿态
找出让机器人最稳定的配置
"""
import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ros2_bridge.mujoco_ros_node import Go2Simulator


# 常见的四足机器人默认姿态
TEST_POSES = {
    "当前EVA02": [0.0, 0.9, -1.6],
    "Go2标准": [0.0, 1.0, -1.5],
    "Go2低姿态": [0.0, 0.8, -1.4],
    "ANYmal": [0.0, 0.5, -1.0],
    "高站立": [0.0, 1.2, -2.0],
    "低站立": [0.0, 0.6, -1.2],
    "中等站立": [0.0, 0.75, -1.3],
}


def test_pose(pose_name, pose, xml_path, model_path, action_scale=0.5, steps=50):
    """测试一个姿态配置"""

    print(f"\n{'='*70}")
    print(f"🧪 测试: {pose_name}")
    print(f"{'='*70}")
    print(f"姿态: hip={pose[0]:.2f}, thigh={pose[1]:.2f}, calf={pose[2]:.2f}")

    # 创建仿真器
    sim = Go2Simulator(xml_path, rl_model_path=model_path, action_scale=action_scale)

    # 覆盖默认姿态
    default_pose_full = np.tile(pose, 4)  # 4条腿
    sim._default_dof_pos = default_pose_full

    # 重新初始化到新姿态
    sim._init_stand()

    # 设置零速度指令
    sim.set_command(0.0, 0.0, 0.0)

    # 记录数据
    base_heights = []
    action_magnitudes = []

    for i in range(steps):
        obs = sim._build_obs()

        # RL推理
        if sim._rl_type == "onnx":
            ort_inputs = {sim._rl_model.get_inputs()[0].name: obs.reshape(1, -1)}
            action_raw = sim._rl_model.run(None, ort_inputs)[0][0]

        action = np.tanh(action_raw)
        action_magnitude = np.abs(action).mean()
        action_magnitudes.append(action_magnitude)

        # 执行
        state = sim.step()
        base_heights.append(state['base_pos'][2])

    # 分析结果
    base_heights = np.array(base_heights)
    action_magnitudes = np.array(action_magnitudes)

    # 计算稳定性指标
    height_stable = base_heights.std() < 0.03  # 高度波动小
    height_ok = base_heights.min() > 0.25      # 没有摔倒
    action_small = action_magnitudes.mean() < 0.5  # 动作不大

    # 评分
    score = 0
    if height_stable:
        score += 40
    if height_ok:
        score += 30
    if action_small:
        score += 30

    print(f"\n📊 结果:")
    print(f"  基座高度: {base_heights.mean():.3f}±{base_heights.std():.3f} m")
    print(f"  高度范围: [{base_heights.min():.3f}, {base_heights.max():.3f}]")
    print(f"  平均动作幅度: {action_magnitudes.mean():.3f}")
    print(f"  动作范围: [{action_magnitudes.min():.3f}, {action_magnitudes.max():.3f}]")

    print(f"\n✅ 评估:")
    print(f"  {'✓' if height_stable else '✗'} 高度稳定 (std < 0.03)")
    print(f"  {'✓' if height_ok else '✗'} 未摔倒 (min > 0.25)")
    print(f"  {'✓' if action_small else '✗'} 动作适中 (mean < 0.5)")
    print(f"  综合得分: {score}/100")

    return {
        'name': pose_name,
        'pose': pose,
        'score': score,
        'height_mean': base_heights.mean(),
        'height_std': base_heights.std(),
        'height_min': base_heights.min(),
        'action_mean': action_magnitudes.mean(),
        'stable': height_stable and height_ok and action_small,
    }


def main():
    xml_path = "/home/tino66/Cs_RL/sim2sim/src/mjcf/eva02/eva02_detailed.xml"
    model_path = "/home/tino66/Cs_RL/sim2sim/model/policy.onnx"

    if not Path(model_path).exists():
        print(f"错误: 模型不存在 {model_path}")
        return

    print("=" * 70)
    print("🔍 自动测试不同的默认姿态")
    print("=" * 70)
    print(f"模型: {model_path}")
    print(f"测试姿态数: {len(TEST_POSES)}")
    print(f"每个姿态测试步数: 50")
    print(f"总耗时约: {len(TEST_POSES) * 3}秒")

    results = []

    for pose_name, pose in TEST_POSES.items():
        try:
            result = test_pose(pose_name, pose, xml_path, model_path)
            results.append(result)
            time.sleep(0.5)  # 短暂暂停
        except Exception as e:
            print(f"✗ 测试失败: {e}")

    # 排序结果
    results.sort(key=lambda x: x['score'], reverse=True)

    # 汇总报告
    print("\n" + "=" * 70)
    print("📊 测试结果汇总 (按得分排序)")
    print("=" * 70)

    print(f"\n{'姿态':<15} {'得分':>6} {'高度':>8} {'波动':>8} {'动作':>8} {'状态':>6}")
    print("-" * 70)

    for r in results:
        status = "✅稳定" if r['stable'] else "❌不稳"
        print(f"{r['name']:<15} {r['score']:>6}/100 "
              f"{r['height_mean']:>7.3f}m {r['height_std']:>7.3f}m "
              f"{r['action_mean']:>7.3f} {status:>8}")

    # 推荐
    print("\n" + "=" * 70)
    print("💡 推荐配置")
    print("=" * 70)

    best = results[0]
    print(f"\n最佳姿态: {best['name']}")
    print(f"配置: hip={best['pose'][0]:.2f}, thigh={best['pose'][1]:.2f}, calf={best['pose'][2]:.2f}")
    print(f"得分: {best['score']}/100")

    if best['score'] < 50:
        print(f"\n⚠️  警告: 所有配置得分都不高")
        print(f"可能的问题:")
        print(f"  1. 训练时使用了完全不同的关节配置")
        print(f"  2. 关节顺序/命名不匹配")
        print(f"  3. 观测缩放因子不匹配")
        print(f"  4. 模型训练未收敛")
    else:
        print(f"\n修改配置文件:")
        print(f"  编辑: sim2sim/config/robot/eva02.yaml")
        print(f"  修改 default_pose 部分为:")
        print(f"    FL_hip: {best['pose'][0]}")
        print(f"    FL_thigh: {best['pose'][1]}")
        print(f"    FL_calf: {best['pose'][2]}")
        print(f"    (其他三条腿相同)")


if __name__ == "__main__":
    main()
