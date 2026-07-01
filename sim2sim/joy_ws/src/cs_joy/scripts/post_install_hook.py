import os
from ament_index_python.packages import get_package_prefix

def fix_executables():
    """构建后钩子：创建 lib/cs_joy 到 bin 的链接"""
    try:
        pkg_prefix = get_package_prefix('cs_joy')
        lib_dir = os.path.join(pkg_prefix, 'lib', 'cs_joy')
        bin_dir = os.path.join(pkg_prefix, 'bin')

        if not os.path.exists(bin_dir):
            return

        os.makedirs(lib_dir, exist_ok=True)

        for exe in os.listdir(bin_dir):
            src = os.path.join('..', '..', 'bin', exe)
            dst = os.path.join(lib_dir, exe)

            if os.path.exists(dst):
                os.remove(dst)

            os.symlink(src, dst)
            print(f"✓ 链接 {exe} 到 lib/cs_joy/")
    except Exception as e:
        print(f"警告: 无法自动修复路径 - {e}")

if __name__ == '__main__':
    fix_executables()
