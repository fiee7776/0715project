#!/usr/bin/env python3
"""测试关节控制是否正常工作"""

import sys
sys.path.insert(0, '.')
from src.simulator import RobotSimulator
import numpy as np

URDF = 'assets/humanoid_robot_mediapipe_v2.urdf'

# 10 个关节：torso_bend, torso_twist,
#   left_shoulder, left_elbow, right_shoulder, right_elbow,
#   left_hip, left_knee, right_hip, right_knee

sim = RobotSimulator(URDF, use_gui=False)
print(f'已加载: {URDF}，控制关节数: {len(sim.joint_ids)}')

print('\n测试 1: 双臂平举 + 站立')
sim.set_joint_angles(np.array([0, 0, 0, 0, 0, 0, -90, 0, -90, 0]))
for _ in range(10):
    sim.step_simulation()
print(f'  关节位置: {sim.get_robot_state()["joint_positions"]}')

print('\n测试 2: 双臂上举 + 弯腰')
sim.set_joint_angles(np.array([20, 0, 90, 0, 90, 0, -90, 0, -90, 0]))
for _ in range(10):
    sim.step_simulation()
print(f'  关节位置: {sim.get_robot_state()["joint_positions"]}')

print('\n测试 3: 还原默认姿态')
sim.set_joint_angles(np.array([0, 0, -90, 0, -90, 0, -90, 0, -90, 0]))
for _ in range(10):
    sim.step_simulation()
print(f'  关节位置: {sim.get_robot_state()["joint_positions"]}')

sim.close()
print('\n✓ 测试完成')
