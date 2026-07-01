#!/bin/bash
################################################################################
# sim2sim.sh - EVA02 完整启动脚本
# 启动 MuJoCo 仿真器 + evdev 键盘控制器
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

################################################################################
# 1. 环境检查
################################################################################

print_info "检查环境..."

# 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "sim2sim" ]; then
    print_warning "未在 sim2sim conda 环境中"
    print_info "激活 conda 环境..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate sim2sim
fi

# 检查 ROS2
if [ -z "$ROS_DISTRO" ]; then
    print_warning "ROS2 环境未加载"
    print_info "加载 ROS2 环境..."
    source /opt/ros/humble/setup.bash
fi

# 进入工作目录
cd "$(dirname "$0")"
WORKSPACE=$(pwd)
print_success "工作目录: $WORKSPACE"

################################################################################
# 2. 清理旧进程
################################################################################

print_info "清理旧进程..."

pkill -f "mujoco_ros_node" 2>/dev/null && print_success "已停止旧的仿真器" || true
pkill -f "keyboard_controller_evdev" 2>/dev/null && print_success "已停止旧的键盘控制器" || true

sleep 1

################################################################################
# 3. 配置参数
################################################################################

# 机器人配置
ROBOT="eva02"
MODEL="model/policy.onnx"
ACTION_SCALE="0.3"

# 日志目录
LOG_DIR="/tmp/sim2sim"
mkdir -p "$LOG_DIR"

print_info "配置参数:"
echo "  机器人: $ROBOT"
echo "  模型: $MODEL"
echo "  action_scale: $ACTION_SCALE"

################################################################################
# 4. 启动 MuJoCo 仿真器
################################################################################

print_info "启动 MuJoCo 仿真器..."

python -m src.ros2_bridge.mujoco_ros_node \
    --robot "$ROBOT" \
    --model "$MODEL" \
    --action-scale "$ACTION_SCALE" \
    > "$LOG_DIR/mujoco.log" 2>&1 &

MUJOCO_PID=$!
print_success "仿真器已启动 (PID: $MUJOCO_PID)"

# 等待仿真器启动
sleep 3

# 检查进程是否还在运行
if ! kill -0 $MUJOCO_PID 2>/dev/null; then
    print_error "仿真器启动失败！"
    print_info "查看日志: cat $LOG_DIR/mujoco.log"
    tail -20 "$LOG_DIR/mujoco.log"
    exit 1
fi

# 检查 ROS2 话题
print_info "检查 ROS2 话题..."
sleep 2

if ros2 topic list | grep -q "/joint_states"; then
    print_success "ROS2 话题正常发布"
    ros2 topic list | grep -E "cmd_vel|joint_states|odom|foot_contacts"
else
    print_error "ROS2 话题未发布！"
    print_info "查看日志: cat $LOG_DIR/mujoco.log"
    exit 1
fi

################################################################################
# 5. 启动键盘控制器 (evdev)
################################################################################

print_info "启动键盘控制器 (evdev)..."

# 进入 ROS2 工作空间
cd joy_ws
source install/setup.bash

# 启动键盘控制器（需要 root 权限访问 /dev/input）
sudo sg input -c "bash -c '\
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate sim2sim && \
    source /opt/ros/humble/setup.bash && \
    source install/setup.bash && \
    python3 src/cs_joy/cs_joy/keyboard_controller_evdev.py \
    > $LOG_DIR/keyboard.log 2>&1'" &

KEYBOARD_PID=$!
cd "$WORKSPACE"

sleep 2

# 检查键盘控制器是否启动
if pgrep -f "keyboard_controller_evdev" > /dev/null; then
    print_success "键盘控制器已启动"
else
    print_warning "键盘控制器可能未启动，请检查日志"
fi

################################################################################
# 6. 显示状态和使用说明
################################################################################

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ sim2sim 已启动${NC}"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "📋 进程信息:"
echo "  仿真器 PID: $MUJOCO_PID"
echo "  键盘控制器: $(pgrep -f keyboard_controller_evdev || echo '未检测到')"
echo ""
echo "📁 日志文件:"
echo "  仿真器: $LOG_DIR/mujoco.log"
echo "  键盘: $LOG_DIR/keyboard.log"
echo ""
echo "🎮 键盘控制:"
echo "  ↑ / W  - 前进"
echo "  ↓ / S  - 后退"
echo "  ← / Q  - 左转"
echo "  → / E  - 右转"
echo "  Space  - 停止"
echo ""
echo "🛠️  调试命令:"
echo "  查看话题: ros2 topic list"
echo "  监控速度: ros2 topic echo /cmd_vel"
echo "  监控关节: ros2 topic echo /joint_states"
echo ""
echo "🛑 停止命令:"
echo "  ./stop_sim2sim.sh"
echo "  或手动: pkill -f mujoco_ros_node && pkill -f keyboard_controller_evdev"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# 保存 PID 到文件，方便后续停止
echo "$MUJOCO_PID" > "$LOG_DIR/mujoco.pid"

print_info "按 Ctrl+C 退出此脚本（进程会继续在后台运行）"
print_info "或运行 ./stop_sim2sim.sh 停止所有进程"

# 可选：持续显示日志
# tail -f "$LOG_DIR/mujoco.log"
