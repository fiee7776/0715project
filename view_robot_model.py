#!/usr/bin/env python3
"""
机器人模型查看器
显示 URDF 机器人模型的结构和外形
"""

import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pybullet as p
import pybullet_data
import numpy as np

def view_robot_model(urdf_path, display_time=10):
    """
    在 PyBullet GUI 中显示机器人模型
    
    Args:
        urdf_path: URDF 文件路径
        display_time: 显示时间（秒）
    """
    # 连接到 PyBullet GUI
    client_id = p.connect(p.GUI)
    print(f"[RobotViewer] 已连接 PyBullet GUI")
    
    # 设置物理参数
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 加载地面
    p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
    print(f"[RobotViewer] 已加载地面")
    
    # 加载机器人模型
    if not Path(urdf_path).exists():
        raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")
    
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 1.0],
        useFixedBase=True
    )
    print(f"[RobotViewer] 已加载机器人模型: {urdf_path}")
    
    # 获取机器人信息
    num_joints = p.getNumJoints(robot_id)
    print(f"\n[机器人结构]")
    print(f"  总关节数: {num_joints}")
    print(f"\n  关节列表:")
    
    all_joint_info = []
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        joint_type = info[2]
        child_link = info[12].decode('utf-8')
        
        type_name = {
            p.JOINT_REVOLUTE: "旋转",
            p.JOINT_PRISMATIC: "平移",
            p.JOINT_FIXED: "固定",
            p.JOINT_SPHERICAL: "球形"
        }.get(joint_type, "其他")
        
        print(f"    [{i:2d}] {joint_name:20s} ({type_name:3s}) -> {child_link}")
        all_joint_info.append({
            'id': i,
            'name': joint_name,
            'type': type_name,
            'child': child_link
        })
    
    # 设置摄像机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.8]
    )
    
    print(f"\n[显示信息]")
    print(f"  鼠标操作: 左键拖拽旋转，右键缩放，中键平移")
    print(f"  显示时间: {display_time} 秒")
    print(f"  按 Ctrl+C 可提前关闭")
    
    # 显示机器人
    start_time = time.time()
    try:
        while time.time() - start_time < display_time:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        print("\n[已中断]")
    
    # 断开连接
    p.disconnect(client_id)
    print(f"\n✓ 查看完成")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='查看机器人模型')
    parser.add_argument('--urdf', type=str, default='assets/humanoid_robot_mediapipe_v2.urdf',
                        help='URDF 模型文件路径')
    parser.add_argument('--time', type=int, default=15,
                        help='显示时间（秒）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("机器人模型查看器 (Robot Model Viewer)")
    print("=" * 60)
    print()
    
    view_robot_model(args.urdf, args.time)
