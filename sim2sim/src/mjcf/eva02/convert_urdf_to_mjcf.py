#!/usr/bin/env python3
"""
将EVA02的URDF转换为MuJoCo MJCF格式
"""
import os
import mujoco

def convert_urdf_to_mjcf():
    """转换URDF到MJCF"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "urdf", "EVA02_simplify_description.urdf")
    output_path = os.path.join(current_dir, "eva02.xml")

    print(f"正在转换 URDF: {urdf_path}")

    # 读取URDF文件内容
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()

    # 修复mesh路径 - 从ROS package路径改为相对路径
    urdf_content = urdf_content.replace(
        'package://EVA02_description/meshes/',
        'meshes/'
    )

    # 保存修改后的URDF到临时文件
    temp_urdf = os.path.join(current_dir, "temp_eva02.urdf")
    with open(temp_urdf, 'w') as f:
        f.write(urdf_content)

    try:
        # 使用MuJoCo加载URDF
        print("正在加载并转换模型...")
        model = mujoco.MjModel.from_xml_path(temp_urdf)

        # 保存为MJCF格式
        mujoco.mj_saveLastXML(output_path, model)
        print(f"✓ 转换成功！输出文件: {output_path}")

        # 清理临时文件
        os.remove(temp_urdf)

        return True

    except Exception as e:
        print(f"✗ 转换失败: {e}")
        print("\n提示: MuJoCo的URDF加载器可能需要一些调整。")
        print("我们将创建一个手动的MJCF文件模板。")
        if os.path.exists(temp_urdf):
            os.remove(temp_urdf)
        return False

if __name__ == "__main__":
    success = convert_urdf_to_mjcf()
    if not success:
        print("\n你可以:")
        print("1. 手动编辑生成的MJCF文件")
        print("2. 使用在线转换工具: https://www.gymlibrary.dev/content/mujoco/")
        print("3. 参考 go2.xml 的结构手动创建")
