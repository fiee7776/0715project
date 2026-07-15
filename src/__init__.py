# Humanoid Dance Project - 具身智能模块包
"""
模块化的具身智能处理框架
负责从视觉捕捉 -> 数据处理 -> 动作映射的完整流程
"""

from .capture import MotionCapturer
from .processor import MotionProcessor
from .mapping import MotionMapper
from .simulator import RobotSimulator

__all__ = ['MotionCapturer', 'MotionProcessor', 'MotionMapper', 'RobotSimulator']
