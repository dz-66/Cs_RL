"""
机器人配置管理器
支持动态加载不同的机器人配置（Go2, EVA02等）
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class RobotConfig:
    """机器人配置类"""

    def __init__(self, config_path: str):
        """
        加载机器人配置文件

        Args:
            config_path: YAML配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.name = self.config['robot']['name']
        self._parse_config()

    def _parse_config(self):
        """解析配置"""
        robot = self.config['robot']

        # 基本信息
        self.description = robot.get('description', '')
        self.mass = robot['body']['mass']
        self.base_height = robot['body']['base_height']

        # 关节名称（根据配置自动推断）
        self.dof_names = self._get_dof_names()

        # 默认姿态
        self.default_pose = self._parse_default_pose(robot.get('default_pose', {}))

        # 电机参数
        self.kp = robot['motors']['kp']
        self.kd = robot['motors']['kd']

        # 步态参数
        if 'gait' in self.config:
            gait = self.config['gait']
            self.gait_frequency = gait.get('frequency', 1.5)
            self.gait_step_height = gait.get('step_height', 0.08)

    def _get_dof_names(self) -> List[str]:
        """
        根据配置推断关节名称
        四足机器人标准顺序: FL, FR, RL, RR
        每条腿: hip, thigh, calf
        """
        legs = ['FL', 'FR', 'RL', 'RR']
        joints = ['hip', 'thigh', 'calf']
        return [f"{leg}_{joint}" for leg in legs for joint in joints]

    def _parse_default_pose(self, pose_dict: Dict) -> np.ndarray:
        """解析默认站立姿态"""
        # Go2使用旧格式，没有default_pose，使用默认值
        if not pose_dict:
            return np.array([
                0.0, 1.0, -1.5,  # FL
                0.0, 1.0, -1.5,  # FR
                0.0, 1.0, -1.5,  # RL
                0.0, 1.0, -1.5,  # RR
            ])

        pose = []
        for leg in ['FL', 'FR', 'RL', 'RR']:
            pose.extend([
                pose_dict[f'{leg}_hip'],
                pose_dict[f'{leg}_thigh'],
                pose_dict[f'{leg}_calf']
            ])
        return np.array(pose)

    def get_actuator_names(self) -> List[str]:
        """获取执行器名称列表"""
        return [f"{name}_motor" for name in self.dof_names]

    def get_joint_names(self) -> List[str]:
        """获取关节名称列表"""
        return [f"{name}_joint" for name in self.dof_names]


class RobotFactory:
    """机器人工厂类，用于创建不同的机器人配置"""

    # 获取sim2sim目录（src的父目录）
    _base_dir = Path(__file__).resolve().parent.parent
    _configs_dir = _base_dir / "config" / "robot"
    _mjcf_dir = _base_dir / "src" / "mjcf"

    _robot_registry = {
        'go2': {
            'config': 'go2.yaml',
            'mjcf': 'go2/go2.xml',
        },
        'eva02': {
            'config': 'eva02.yaml',
            'mjcf': 'eva02/eva02_detailed.xml',  # 使用详细网格版本
        }
    }

    @classmethod
    def list_robots(cls) -> List[str]:
        """列出所有可用的机器人"""
        return list(cls._robot_registry.keys())

    @classmethod
    def create(cls, robot_name: str) -> tuple[RobotConfig, str]:
        """
        创建机器人配置和模型路径

        Args:
            robot_name: 机器人名称 (go2, eva02等)

        Returns:
            (RobotConfig对象, MJCF模型路径)
        """
        robot_name = robot_name.lower()

        if robot_name not in cls._robot_registry:
            available = ', '.join(cls.list_robots())
            raise ValueError(
                f"未知的机器人: {robot_name}\n"
                f"可用的机器人: {available}"
            )

        robot_info = cls._robot_registry[robot_name]

        # 配置文件路径
        config_path = cls._configs_dir / robot_info['config']
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # MJCF模型路径
        mjcf_path = cls._mjcf_dir / robot_info['mjcf']
        if not mjcf_path.exists():
            raise FileNotFoundError(f"MJCF模型不存在: {mjcf_path}")

        # 创建配置对象
        config = RobotConfig(str(config_path))

        return config, str(mjcf_path)

    @classmethod
    def get_mjcf_path(cls, robot_name: str) -> str:
        """获取机器人的MJCF模型路径"""
        robot_name = robot_name.lower()
        if robot_name not in cls._robot_registry:
            raise ValueError(f"未知的机器人: {robot_name}")

        mjcf_path = cls._mjcf_dir / cls._robot_registry[robot_name]['mjcf']
        return str(mjcf_path)


if __name__ == "__main__":
    # 测试
    print("可用的机器人:", RobotFactory.list_robots())

    for robot_name in RobotFactory.list_robots():
        print(f"\n=== {robot_name.upper()} ===")
        try:
            config, mjcf_path = RobotFactory.create(robot_name)
            print(f"名称: {config.name}")
            print(f"描述: {config.description}")
            print(f"质量: {config.mass} kg")
            print(f"基座高度: {config.base_height} m")
            print(f"关节数: {len(config.dof_names)}")
            print(f"MJCF路径: {mjcf_path}")
            print(f"默认姿态形状: {config.default_pose.shape}")
        except Exception as e:
            print(f"错误: {e}")
