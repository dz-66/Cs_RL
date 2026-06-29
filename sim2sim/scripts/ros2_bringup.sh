#!/bin/bash
# ============================================
# ROS2 + MuJoCo 桥接启动脚本
# 
# 用法:
#   ./scripts/ros2_bringup.sh              # 键盘控制模式
#   ./scripts/ros2_bringup.sh --joy        # 手柄控制模式
#   ./scripts/ros2_bringup.sh --standalone # 独立模式 (无ROS2)
# ============================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODE="keyboard"

while [[ $# -gt 0 ]]; do
    case $1 in
        --joy)
            MODE="joy"
            shift
            ;;
        --keyboard)
            MODE="keyboard"
            shift
            ;;
        --standalone)
            MODE="standalone"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  Go2 MuJoCo + ROS2 桥接"
echo "  控制模式: $MODE"
echo "=========================================="

case $MODE in
    standalone)
        echo "启动独立模式 (键盘控制 + 渲染窗口)..."
        cd "$PROJECT_DIR"
        python -m src.ros2_bridge.mujoco_ros_node --standalone
        ;;
    
    keyboard)
        echo "启动 ROS2 键盘控制模式..."
        echo ""
        echo "请在不同终端中运行:"
        echo ""
        echo "  终端1: cd $PROJECT_DIR && python -m src.ros2_bridge.mujoco_ros_node"
        echo "  终端2: ros2 run cs_joy keyboard_controller"
        echo ""
        
        # 自动启动
        cd "$PROJECT_DIR"
        python -m src.ros2_bridge.mujoco_ros_node &
        SIM_PID=$!
        
        sleep 1
        
        ros2 run cs_joy keyboard_controller &
        KB_PID=$!
        
        echo "仿真 PID: $SIM_PID, 键盘 PID: $KB_PID"
        echo "按 Ctrl+C 停止..."
        
        trap "kill $SIM_PID $KB_PID 2>/dev/null; exit" INT TERM
        wait
        ;;
    
    joy)
        echo "启动 ROS2 手柄控制模式..."
        echo ""
        
        # 启动手柄驱动
        echo "正在启用手柄驱动 (joy_node)..."
        ros2 run joy joy_node &
        JOY_PID=$!
        sleep 1
        
        # 启动手柄控制器
        ros2 run cs_joy joy_controller &
        CTRL_PID=$!
        
        # 启动仿真
        cd "$PROJECT_DIR"
        python -m src.ros2_bridge.mujoco_ros_node &
        SIM_PID=$!
        
        echo "手柄驱动 PID: $JOY_PID"
        echo "手柄控制器 PID: $CTRL_PID"
        echo "仿真 PID: $SIM_PID"
        echo "按住 LB 按钮开始控制..."
        
        trap "kill $SIM_PID $CTRL_PID $JOY_PID 2>/dev/null; exit" INT TERM
        wait
        ;;
esac
