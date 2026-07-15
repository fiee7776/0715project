"""
虚拟机器人仿真脚本 (Robot Simulation Script)
加载关节数据并在 PyBullet 虚拟环境中播放机器人动作
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# 添加 src 模块到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from simulator import RobotSimulator


def main():
    parser = argparse.ArgumentParser(description='虚拟机器人动作仿真')
    parser.add_argument('--csv_file', type=str, default='outputs/joint_sequence.csv',
                        help='关节数据 CSV 文件路径')
    parser.add_argument('--urdf_file', type=str, default='assets/humanoid_robot_mediapipe_v2.urdf',
                        help='机器人 URDF 模型文件路径')
    parser.add_argument('--frame_rate', type=float, default=30.0,
                        help='视频帧率')
    parser.add_argument('--loop', action='store_true',
                        help='循环播放动作')
    parser.add_argument('--headless', action='store_true',
                        help='无 GUI 模式（仅计算）')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='最多播放的帧数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("虚拟机器人动作仿真系统")
    print("=" * 60)
    
    # 验证文件存在
    if not Path(args.csv_file).exists():
        print(f"✗ 错误: 关节数据文件不存在: {args.csv_file}")
        return
    
    if not Path(args.urdf_file).exists():
        print(f"✗ 错误: URDF 文件不存在: {args.urdf_file}")
        return
    
    print(f"\n[参数配置]")
    print(f"  关节数据: {args.csv_file}")
    print(f"  机器人模型: {args.urdf_file}")
    print(f"  帧率: {args.frame_rate} FPS")
    print(f"  循环播放: {'是' if args.loop else '否'}")
    print(f"  GUI 模式: {'无' if args.headless else '有'}")
    if args.max_frames:
        print(f"  最多帧数: {args.max_frames}")
    
    # 创建仿真环境
    try:
        with RobotSimulator(
            urdf_path=args.urdf_file,
            use_gui=not args.headless
        ) as simulator:
            
            # 加载关节数据
            print(f"\n[步骤 1] 加载关节数据...")
            joint_sequence = simulator.load_joint_data_from_csv(args.csv_file)
            
            # 播放动作
            print(f"\n[步骤 2] 播放机器人动作...")
            print("  提示: 在 GUI 中可以用鼠标旋转、缩放视角")
            print("  提示: 按 Ctrl+C 可以暂停播放\n")
            
            # 调整摄像机
            simulator.set_camera(distance=3.0, yaw=45, pitch=-30)
            
            # 播放序列
            simulator.play_joint_sequence(
                joint_sequence,
                frame_rate=args.frame_rate,
                loop=args.loop,
                max_frames=args.max_frames
            )
            
            print(f"\n[步骤 3] 仿真完成")
            
            # 最后显示机器人的最终状态
            final_state = simulator.get_robot_state()
            print(f"\n[最终状态]")
            print(f"  基座位置: {final_state['base_position']}")
            print(f"  关节位置数: {len(final_state['joint_positions'])}")
            print(f"  模拟时间: {final_state['simulation_time']:.2f} 秒")
    
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
