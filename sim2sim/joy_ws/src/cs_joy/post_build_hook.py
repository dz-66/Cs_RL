# Colcon 构建后钩子
# 自动创建 lib/cs_joy 符号链接

import os
import sys

def main():
    """colcon 构建后自动修复 ROS2 可执行文件路径"""
    # 从环境变量获取安装路径
    install_base = os.environ.get('COLCON_PREFIX_PATH', '').split(':')[0]

    if not install_base:
        # 回退到相对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        install_base = os.path.join(script_dir, '..', '..', 'install', 'cs_joy')

    lib_dir = os.path.join(install_base, 'lib', 'cs_joy')
    bin_dir = os.path.join(install_base, 'bin')

    if not os.path.exists(bin_dir):
        print(f"[post-build] bin/ 目录不存在: {bin_dir}")
        return

    # 创建 lib/cs_joy
    os.makedirs(lib_dir, exist_ok=True)

    # 创建符号链接
    for exe in os.listdir(bin_dir):
        src = os.path.join('..', '..', 'bin', exe)
        dst = os.path.join(lib_dir, exe)

        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)

        os.symlink(src, dst)
        print(f"[post-build] ✓ 链接 {exe} -> lib/cs_joy/")

if __name__ == '__main__':
    main()
