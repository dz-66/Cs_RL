from setuptools import setup

package_name = "cs_joy"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="user@example.com",
    description="Go2 四足机器人手柄/键盘控制包",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "joy_controller = cs_joy.joy_controller:main",
            "keyboard_controller = cs_joy.keyboard_controller:main",
            "mujoco_ros_node = cs_joy.mujoco_ros_node:main",
        ],
    },
)
