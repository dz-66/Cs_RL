# RL模型部署故障排查指南

## 📊 当前状态

根据测试结果，你的机器人**实际上是稳定的**：
- ✅ 基座高度保持在 0.29-0.40m（合理范围）
- ✅ 100步仿真未摔倒
- ✅ 观测输入45维正确
- ✅ 已启用tanh激活函数

## ⚠️ 可能的问题

如果你观察到"站不稳"的现象，可能是以下原因：

### 1. **动作缩放不匹配** ⭐ 最常见

**症状**：机器人抖动、腿部动作过大/过小

**当前配置**：`action_scale = 0.25`

**解决方法**：尝试不同的值
```bash
# 保守（小动作）
python -m src.ros2_bridge.mujoco_ros_node --standalone --robot eva02 \
    --model model/policy.onnx --action-scale 0.25

# 标准
python -m src.ros2_bridge.mujoco_ros_node --standalone --robot eva02 \
    --model model/policy.onnx --action-scale 0.5

# 激进（大动作）
python -m src.ros2_bridge.mujoco_ros_node --standalone --robot eva02 \
    --model model/policy.onnx --action-scale 1.0
```

### 2. **默认姿态不匹配**

**当前配置**：`[0.0, 0.9, -1.6] × 4腿`

**检查方法**：
```bash
# 查看训练配置中的 default_dof_pos
# 通常在 legged_gym/envs/*/config.py 中
```

**修改位置**：`sim2sim/config/robot/eva02.yaml`

### 3. **观测缩放因子不匹配**

**当前配置**：
```python
ang_vel_scale = 0.25
dof_vel_scale = 0.05
commands_scale = [2.0, 2.0, 0.25]
```

**修改位置**：`sim2sim/src/ros2_bridge/mujoco_ros_node.py:148-154`

### 4. **模型训练未收敛**

**检查方法**：
- 查看训练日志中的reward曲线
- 在训练环境（Isaac Gym/Lab）中测试相同的ONNX模型
- 如果训练环境也不稳定，需要重新训练

### 5. **控制频率不匹配**

**当前配置**：
- 仿真步长：0.005s (200Hz)
- Decimation：4
- 控制频率：50Hz

**修改位置**：`sim2sim/src/ros2_bridge/mujoco_ros_node.py:167-169`

### 6. **PD增益不匹配**

**当前配置**（来自MJCF）：
```xml
kp="100.0" kv="5.0"
```

**修改位置**：`sim2sim/src/mjcf/eva02/eva02_detailed.xml`

## 🔧 调试步骤

### Step 1: 确认模型输入输出形状
```bash
python -c "import onnxruntime as ort; \
    sess = ort.InferenceSession('model/policy.onnx'); \
    print('Input:', sess.get_inputs()[0].shape); \
    print('Output:', sess.get_outputs()[0].shape)"
```
应该输出：`Input: [1, 45]`, `Output: [1, 12]`

### Step 2: 运行观测诊断
```bash
python scripts/diagnose_obs.py --robot eva02 --steps 50
```
检查观测向量的各个分量是否合理

### Step 3: 测试RL推理
```bash
python scripts/test_rl_inference.py --robot eva02 --steps 100
```
观察基座高度是否稳定

### Step 4: 尝试不同的action_scale
```bash
for scale in 0.25 0.5 1.0; do
    echo "Testing action_scale=$scale"
    python -m src.ros2_bridge.mujoco_ros_node --standalone --robot eva02 \
        --model model/policy.onnx --action-scale $scale
done
```

### Step 5: 对比训练配置
```bash
python scripts/compare_configs.py
```
填写检查清单，逐项对比

## 🎯 快速解决方案

如果你只是想快速验证模型，试试这个：

```bash
# 最可能正确的配置
python -m src.ros2_bridge.mujoco_ros_node --standalone --robot eva02 \
    --model model/policy.onnx --action-scale 0.5
```

使用键盘控制：
- ↑/↓ 前进/后退
- ←/→ 左转/右转
- Space 停止
- Esc 退出

## 📝 已修复的问题

- ✅ 启用了tanh激活函数（`mujoco_ros_node.py:317`）
- ✅ 观测向量构建正确（45维）
- ✅ 关节命名自动检测（EVA02 vs Go2）
- ✅ 默认姿态配置（从YAML加载）

## 📚 相关文件

- **配置文件**：`sim2sim/config/robot/eva02.yaml`
- **主控制器**：`sim2sim/src/ros2_bridge/mujoco_ros_node.py`
- **MJCF模型**：`sim2sim/src/mjcf/eva02/eva02_detailed.xml`
- **诊断工具**：`sim2sim/scripts/diagnose_obs.py`

## 💡 进阶调试

如果以上方法都不行，考虑：

1. **保存训练时的观测样本**
   ```python
   # 在训练代码中
   np.save('train_obs_sample.npy', obs)
   ```

2. **在sim2sim中加载并对比**
   ```python
   train_obs = np.load('train_obs_sample.npy')
   sim_obs = sim._build_obs()
   diff = np.abs(train_obs - sim_obs)
   print(f"差异: {diff.max():.3f}")
   ```

3. **检查MJCF物理参数**
   - 质量/惯性是否匹配
   - 摩擦系数是否合理
   - 关节限位是否正确

## 🐛 常见错误

| 症状 | 可能原因 | 解决方法 |
|------|---------|---------|
| 机器人直接摔倒 | action_scale过大 | 降低到0.25 |
| 机器人僵硬不动 | action_scale过小或模型未激活 | 提高到0.5-1.0 |
| 机器人抖动 | PD增益过高或观测噪声 | 检查Kp/Kd |
| 腿部动作异常 | 默认姿态不匹配 | 检查default_dof_pos |
| 模型输出全零 | 观测输入异常 | 运行diagnose_obs.py |

## 📞 需要帮助？

如果问题仍未解决，请提供：
1. 训练配置文件（cfg.py）
2. 训练日志（reward曲线截图）
3. 运行 `test_rl_inference.py` 的完整输出
4. 视频录屏（如果可能）
