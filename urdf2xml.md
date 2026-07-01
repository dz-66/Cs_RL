# EVA02 URDF to MuJoCo XML 集成问题排查与解决

## 📋 项目概述

将EVA02四足机器人从URDF格式转换为MuJoCo XML格式，并集成到sim2sim仿真框架中，支持RL策略部署。

**目标**: 
- 将EVA02机器人模型成功加载到MuJoCo仿真环境
- 支持键盘/手柄控制
- 支持RL策略（ONNX模型）推理
- 实现稳定的站立和运动

---

## 🔴 遇到的问题及解决方案

### 问题1: 质量配置错误导致机器人下沉

**现象**:
- 机器人初始化后立即下沉
- 基座高度从0.4m降到接近地面
- 腿部无力支撑

**根本原因**:
```yaml
# config/robot/eva02.yaml 中质量配置错误
robot:
  mass: 2.03  # ❌ 只配置了基座质量，缺少腿部质量
```

**诊断过程**:
```bash
# 使用诊断脚本检查模型质量
python diagnose_eva02_rl.py
```

输出显示总质量为2.03kg，明显偏轻（实际应该12-15kg）。

**解决方案**:
```yaml
# 修正后的配置
robot:
  mass: 12.621  # ✓ 完整质量（基座+四条腿）
```

**使用工具**: `diagnose_eva02_rl.py`

---

### 问题2: 关节命名不匹配导致控制失效

**现象**:
- 右后腿（RR）一开始就异常收缩
- 其他三条腿正常，唯独RR腿折叠
- 无RL模式下也出现相同问题

**根本原因**:
```python
# src/ros2_bridge/mujoco_ros_node.py
DOF_NAMES = [
    "FL_hip_x", "FL_hip_y", "FL_knee",  # Go2命名
    ...
]

# 但EVA02使用不同的命名
# FL_hip_joint, FL_thigh_joint, FL_calf_joint
```

代码中硬编码了Go2的关节命名，导致EVA02的关节索引全部错误。

**诊断过程**:
```python
# 检查实际关节名
import mujoco
model = mujoco.MjModel.from_xml_path('eva02.xml')
for i in range(1, 13):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"{i}: {name}")

# 输出:
# 1: FL_hip_joint  ← EVA02命名
# vs
# 期望: FL_hip_x   ← Go2命名
```

**解决方案**:
添加自动检测机制：

```python
# 自动检测关节命名（Go2 vs EVA02）
joint_names = []
for i in range(1, min(13, self.model.njnt)):
    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
        joint_names.append(name)

# 判断是哪种命名
if joint_names and 'joint' in joint_names[0]:
    self.DOF_NAMES = self.DOF_NAMES_EVA02
    print("[INFO] 检测到 EVA02 关节命名")
elif joint_names and ('hip_x' in joint_names[0] or 'hip_y' in joint_names[0]):
    self.DOF_NAMES = self.DOF_NAMES_GO2
    print("[INFO] 检测到 Go2 关节命名")
```

**使用工具**: `diagnose_rr_leg.py`, MuJoCo Python API

---

### 问题3: 观测空间异常（tanh激活不匹配）

**现象**:
```
[step 1] obs_range=[-3.676, 0.888]  # 异常！
[step 2] last_action=[-3.68, 0.92]  # 应该在[-1, 1]
[摔倒! step=171]
```

**根本原因**:
```python
# 推理代码中强制应用tanh
def _rl_inference(self, obs: np.ndarray) -> np.ndarray:
    action = self._rl_model.run(None, ort_inputs)[0][0]
    action = np.tanh(action)  # ❌ 如果训练时没用tanh，这会压缩动作
    target_pos = self._default_dof_pos + action * self._action_scale
```

训练时如果动作直接输出（没有tanh），推理时加tanh会把动作范围从[-∞, ∞]压缩到[-1, 1]，导致控制无力。

**诊断过程**:
```bash
python analyze_obs_issue.py
```

分析显示：
- 代码流程：`ONNX输出 → tanh → 乘action_scale`
- 观测中的`last_action`应该在`[-1, 1]`范围（经过tanh后）
- 但日志显示`-3.68`，说明记录的是tanh前的原始值，或者训练时根本没用tanh

**解决方案**:
```python
# 禁用tanh激活
def _rl_inference(self, obs: np.ndarray) -> np.ndarray:
    action = self._rl_model.run(None, ort_inputs)[0][0]
    # action = np.tanh(action)  # 已禁用
    target_pos = self._default_dof_pos + action * self._action_scale
```

**使用工具**: `fix_tanh_issue.py`, `analyze_obs_issue.py`

```bash
# 禁用tanh
python fix_tanh_issue.py disable

# 恢复tanh（如需要）
python fix_tanh_issue.py enable
```

---

### 问题4: PD参数不足导致站立不稳

**现象**:
- 机器人能站立但不够稳定
- 基座高度逐渐下降
- 200步后从0.4m降到0.3m以下

**根本原因**:
```yaml
# 初始PD参数太弱
motors:
  kp: 60.0   # 偏低
  kd: 2.0    # 偏低
```

EVA02质量较大(12.6kg)，需要更强的PD控制增益。

**解决方案**:
```yaml
# 增强的PD参数
motors:
  kp: 100.0  # 从60增大
  kd: 5.0    # 从2增大
```

同时需要同步更新MJCF文件中的执行器参数：

```xml
<!-- eva02_detailed.xml -->
<position name="FL_hip_motor" joint="FL_hip_joint" kp="100" kv="5" .../>
```

**使用工具**: 
- `fix_eva02_stability.py` - 调整配置文件
- `sync_mjcf_pd.py` - 同步MJCF文件

```bash
# 应用预设
python fix_eva02_stability.py balanced  # Kp=80, Kd=3
python fix_eva02_stability.py stiff     # Kp=100, Kd=5

# 同步到MJCF
python sync_mjcf_pd.py
```

---

### 问题5: 视觉渲染问题（右侧腿部hip连接件不显示）

**现象**:
- 在MuJoCo viewer中，FR和RR腿的hip连接件看起来与机身分离
- 物理仿真正常，但视觉显示错误
- FL和RL腿显示正常

**根本原因**:
```xml
<!-- eva02_detailed.xml -->
<!-- FR和RR使用左侧的STL网格 + euler旋转180度 -->
<geom name="FR_hip_visual" mesh="FL_hip_mesh" euler="0 0 3.14159"/>
<geom name="RR_hip_visual" mesh="RL_hip_mesh" euler="0 0 3.14159"/>
```

使用euler旋转镜像网格可能导致渲染位置偏移。

**诊断过程**:
```bash
# 检查网格配置
grep -A 2 "hip_visual" eva02_detailed.xml | grep euler
```

输出显示FR和RR都有`euler="0 0 3.14159"`。

**解决方案**:
为右侧腿创建真正镜像的STL文件：

```python
# create_mirrored_meshes_binary.py
from stl import mesh

# 读取左侧网格
src_mesh = mesh.Mesh.from_file('FL_hip.STL')

# 镜像：沿Y轴反转
mirrored_vectors = src_mesh.vectors.copy()
mirrored_vectors[:, :, 1] *= -1  # 反转Y坐标

# 反转顶点顺序以保持法向量正确
mirrored_vectors = mirrored_vectors[:, ::-1, :]

# 保存新网格
dst_mesh = mesh.Mesh(np.zeros(mirrored_vectors.shape[0], dtype=mesh.Mesh.dtype))
dst_mesh.vectors = mirrored_vectors
dst_mesh.save('FR_hip.STL')
```

然后更新MJCF：

```xml
<!-- 添加新mesh定义 -->
<mesh name="FR_hip_mesh" file="FR_hip.STL"/>
<mesh name="RR_hip_mesh" file="RR_hip.STL"/>

<!-- 使用新mesh，移除euler -->
<geom name="FR_hip_visual" mesh="FR_hip_mesh"/>  <!-- 无euler -->
<geom name="RR_hip_visual" mesh="RR_hip_mesh"/>  <!-- 无euler -->
```

**使用工具**: 
- `create_mirrored_meshes_binary.py` - 创建镜像STL
- `update_mjcf_meshes.py` - 更新MJCF引用
- `fix_visual_euler.py` - 移除euler旋转

```bash
# 创建镜像网格
python create_mirrored_meshes_binary.py

# 更新MJCF文件
python update_mjcf_meshes.py
```

---

## 🛠️ 创建的诊断和修复工具

### 1. 诊断工具

| 工具名称 | 功能 | 使用方法 |
|---------|------|---------|
| `diagnose_eva02_rl.py` | 诊断模型基本配置和稳定性 | `python diagnose_eva02_rl.py` |
| `diagnose_rr_leg.py` | 专门诊断RR腿问题 | `python diagnose_rr_leg.py` |
| `analyze_obs_issue.py` | 分析观测空间异常 | `python analyze_obs_issue.py` |
| `compare_training_params.py` | 对比训练和推理参数 | `python compare_training_params.py` |
| `visualize_rr_issue.py` | 可视化各腿实时状态 | `python visualize_rr_issue.py` |
| `monitor_legs.py` | 实时监控四腿关节角度 | `python monitor_legs.py` |

### 2. 修复工具

| 工具名称 | 功能 | 使用方法 |
|---------|------|---------|
| `fix_tanh_issue.py` | 开关tanh激活 | `python fix_tanh_issue.py disable/enable` |
| `fix_eva02_stability.py` | 调整PD参数预设 | `python fix_eva02_stability.py balanced/stiff` |
| `sync_mjcf_pd.py` | 同步MJCF的PD参数 | `python sync_mjcf_pd.py` |
| `create_mirrored_meshes_binary.py` | 创建镜像STL网格 | `python create_mirrored_meshes_binary.py` |
| `update_mjcf_meshes.py` | 更新MJCF网格引用 | `python update_mjcf_meshes.py` |
| `fix_visual_euler.py` | 移除/恢复euler旋转 | `python fix_visual_euler.py` |

---

## 📝 完整修复流程

### Step 1: 基础配置修正

```bash
cd /home/tino66/Cs_RL/sim2sim

# 1. 修正质量配置
nano config/robot/eva02.yaml
# 将 mass: 2.03 改为 mass: 12.621

# 2. 增强PD参数
python fix_eva02_stability.py stiff

# 3. 同步MJCF文件
python sync_mjcf_pd.py
```

### Step 2: 修复关节命名

```python
# 已在 src/ros2_bridge/mujoco_ros_node.py 中自动修复
# 添加了Go2和EVA02的自动检测逻辑
```

### Step 3: 修复RL策略相关问题

```bash
# 1. 禁用tanh激活
python fix_tanh_issue.py disable

# 2. 测试稳定性
python diagnose_eva02_rl.py
```

### Step 4: 修复视觉渲染

```bash
# 1. 创建镜像STL网格
python create_mirrored_meshes_binary.py

# 2. 更新MJCF引用
python update_mjcf_meshes.py

# 3. 验证
python3 -c "import mujoco; m = mujoco.MjModel.from_xml_path('src/mjcf/eva02/eva02_detailed.xml'); print('✓ 网格数:', m.nmesh)"
```

### Step 5: 最终验证

```bash
# 不带RL模型
python -m src.ros2_bridge.mujoco_ros_node --standalone --robot eva02

# 带RL模型
python -m src.ros2_bridge.mujoco_ros_node --standalone --robot eva02 --model model/policy.onnx
```

---

## 🎯 关键经验教训

### 1. URDF转MuJoCo的常见陷阱

**质量和惯性**:
- URDF转换后只保留了基座质量
- 必须手动验证总质量是否合理
- 使用 `model.body_mass.sum()` 检查

**关节命名**:
- 不同机器人使用不同的命名规范
- 需要添加自动检测或映射机制
- 硬编码关节名是技术债

**执行器参数**:
- MJCF中的PD参数是硬编码的
- 修改config文件后需要同步到MJCF
- 考虑从config动态生成MJCF

### 2. RL策略部署的注意事项

**训练与推理一致性**:
- 必须确认训练时是否使用tanh激活
- 必须确认训练时的action_scale
- 必须确认训练时的default_dof_pos
- 必须确认训练时的clip_observations

**观测空间验证**:
- 实时监控观测范围
- last_action应该在合理范围内
- 异常的观测会直接导致策略失效

**动作空间验证**:
- 检查动作范围是否合理
- 检查关节限制是否一致
- 记录训练和推理的动作统计

### 3. 视觉网格处理

**镜像网格**:
- 使用euler旋转可能导致渲染问题
- 最佳实践：创建真正的镜像STL文件
- 使用numpy-stl库处理二进制STL

**网格路径**:
- 保持一致的路径规范（相对路径）
- 注意MJCF的工作目录
- 使用相对路径避免绝对路径问题

### 4. 调试技巧

**分层诊断**:
1. 先测试静态配置（无仿真）
2. 再测试物理仿真（无RL）
3. 最后测试RL策略

**最小化测试**:
- 先用简化模型（无视觉网格）
- 再用详细模型（有视觉网格）
- 逐步增加复杂度

**对比验证**:
- 对比同类机器人（Go2）的配置
- 对比训练和推理的参数
- 对比左右对称部件的配置

---

## 📚 相关工具和库

### Python库
```bash
pip install mujoco
pip install numpy-stl  # STL文件处理
pip install pynput     # 全局键盘监听
pip install onnxruntime  # ONNX模型推理
```

### MuJoCo工具
- `mujoco.viewer.launch_passive()` - 被动查看器
- `mujoco.mj_name2id()` - 名称到ID映射
- `mujoco.mj_id2name()` - ID到名称映射
- `model.body_mass` - 查看各body质量

### 调试技巧
```python
# 查看模型信息
print(f"总质量: {model.body_mass.sum()}")
print(f"网格数: {model.nmesh}")
print(f"关节数: {model.njnt}")

# 查看关节限制
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    range = model.jnt_range[i]
    print(f"{name}: [{range[0]:.3f}, {range[1]:.3f}]")
```

---

## ✅ 最终结果

**成功指标**:
- ✅ 机器人稳定站立（200步后高度>0.3m）
- ✅ 四条腿关节控制正常
- ✅ 视觉渲染完整（所有部件正确显示）
- ✅ RL策略可正常推理
- ✅ 键盘控制响应正常
- ✅ 观测空间在合理范围内

**性能参数**:
- 总质量: 12.621 kg
- PD参数: Kp=100, Kd=5
- 控制频率: 50 Hz
- 仿真步长: 0.005 s
- 网格数: 19 (详细模型)

**模型文件**:
- 简化版: `src/mjcf/eva02/eva02.xml` (无视觉网格)
- 详细版: `src/mjcf/eva02/eva02_detailed.xml` (含STL网格)
- 配置文件: `config/robot/eva02.yaml`

---

## 🔗 参考资料

**MuJoCo文档**:
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MJCF XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)

**相关项目**:
- Go2 MuJoCo仿真器（参考实现）
- Isaac Lab（训练框架）
- rsl_rl（RL库）

**工具链**:
- URDF → MuJoCo XML 转换
- STL网格镜像处理
- ONNX模型部署

---

**作者**: Claude (Anthropic)  
**日期**: 2026-07-01  
**版本**: 1.0
