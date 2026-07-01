#!/bin/bash
# ROS2包构建后自动修复脚本
# 在 install/cs_joy/lib/cs_joy 创建到 bin 的符号链接

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LIB_DIR="install/cs_joy/lib/cs_joy"
BIN_DIR="install/cs_joy/bin"

if [ ! -d "$BIN_DIR" ]; then
    echo "错误: $BIN_DIR 不存在，请先运行 colcon build"
    exit 1
fi

echo "修复 ROS2 可执行文件路径..."

# 创建 lib/cs_joy 目录
rm -rf "$LIB_DIR"
mkdir -p "$LIB_DIR"

# 创建符号链接
cd "$LIB_DIR"
ln -s ../../bin/* .

echo "✅ 完成！创建的符号链接:"
ls -l

echo ""
echo "验证 ros2 run 可用性:"
cd "$SCRIPT_DIR"
source install/setup.bash
ros2 pkg executables cs_joy
