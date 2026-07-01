#!/bin/bash
################################################################################
# stop_sim2sim.sh - 停止所有 sim2sim 相关进程
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}停止 sim2sim 进程...${NC}"

# 停止仿真器
if pkill -f "mujoco_ros_node"; then
    echo "✓ 已停止 MuJoCo 仿真器"
else
    echo "- 没有运行中的仿真器"
fi

# 停止键盘控制器
if pkill -f "keyboard_controller_evdev"; then
    echo "✓ 已停止键盘控制器"
else
    echo "- 没有运行中的键盘控制器"
fi

# 清理 PID 文件
rm -f /tmp/sim2sim/mujoco.pid

echo ""
echo -e "${GREEN}✓ 所有进程已停止${NC}"
