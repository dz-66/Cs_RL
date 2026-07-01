#!/bin/bash
# Colcon 构建包装脚本
# 自动在构建后修复 ROS2 可执行文件路径

cd "$(dirname "$0")"

# 运行 colcon build
colcon build "$@"

BUILD_STATUS=$?

# 如果构建成功，运行修复脚本
if [ $BUILD_STATUS -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  修复 ROS2 可执行文件路径"
    echo "=========================================="

    # 运行 Python 钩子
    python3 src/cs_joy/post_build_hook.py

    echo ""
    echo "验证 ros2 run 可用性:"
    source install/setup.bash
    ros2 pkg executables cs_joy
fi

exit $BUILD_STATUS
