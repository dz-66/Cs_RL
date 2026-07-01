#!/bin/bash
# 一键构建 ROS2 包并自动修复路径问题

cd "$(dirname "$0")"

echo "=========================================="
echo "  构建 cs_joy ROS2 包"
echo "=========================================="

# 构建
colcon build --symlink-install

if [ $? -ne 0 ]; then
    echo "❌ 构建失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "  修复 ROS2 可执行文件路径"
echo "=========================================="

# 运行修复脚本
./fix_ros2_paths.sh

echo ""
echo "=========================================="
echo "✅ 构建完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  source install/setup.bash"
echo "  ros2 run cs_joy keyboard_controller"
echo ""
echo "或使用快捷脚本:"
echo "  cd /home/tino66/Cs_RL/sim2sim"
echo "  ./run_keyboard_controller.sh"
echo "=========================================="
