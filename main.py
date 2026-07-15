"""
主程序入口 (Main Entry Point)
协调视觉捕捉 -> 数据处理 -> 动作映射的完整流程
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# 添加 src 模块到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from capture import MotionCapturer
from processor import MotionProcessor
from mapping import MotionMapper


def main():
    parser = argparse.ArgumentParser(description='具身智能运动捕捉与映射系统')
    parser.add_argument('--mode', choices=['camera', 'video'], default='camera',
                        help='输入模式：摄像头或视频文件')
    parser.add_argument('--video_path', type=str, default=None,
                        help='视频文件路径（mode=video时必需）')
    parser.add_argument('--output_video', type=str, default=None,
                        help='输出标注视频路径')
    parser.add_argument('--output_joint_csv', type=str, default=None,
                        help='输出关节序列 CSV 文件路径')
    parser.add_argument('--model_complexity', type=int, choices=[0, 1, 2], default=1,
                        help='MediaPipe 模型复杂度 (0=轻, 1=中, 2=重)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("具身智能运动捕捉与映射系统")
    print("=" * 60)
    
    # 步骤 1: 初始化各个模块
    print("\n[步骤 1] 初始化模块...")
    capturer = MotionCapturer(model_complexity=args.model_complexity)
    processor = MotionProcessor(smoothing_window=7, use_cuda=True)
    mapper = MotionMapper()
    
    print(f"✓ 捕捉器已初始化")
    print(f"✓ 处理器已初始化，使用设备: {processor.device}")
    print(f"✓ 映射器已初始化")
    print(f"✓ 映射信息: {mapper.get_mapping_info()}")
    
    # 步骤 2: 数据采集
    print("\n[步骤 2] 开始采集运动数据...")
    
    if args.mode == 'video':
        if not args.video_path:
            print("错误: 视频模式必须指定 --video_path")
            return
        
        if not Path(args.video_path).exists():
            print(f"错误: 视频文件不存在: {args.video_path}")
            return
        
        print(f"从视频文件采集: {args.video_path}")
        all_landmarks = capturer.process_video_file(
            args.video_path,
            output_path=args.output_video
        )
        print(f"✓ 采集完成: {len(all_landmarks)} 帧")
        
    else:  # camera mode
        print("从摄像头采集（按 'q' 退出）...")
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return
        
        all_landmarks = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = capturer.get_landmarks_from_frame(frame)
                all_landmarks.append(landmarks)
                
                # 绘制并显示
                annotated = capturer.draw_landmarks_on_frame(frame, landmarks)
                cv2.imshow('Motion Capture', annotated)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"已采集 {frame_count} 帧...")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"✓ 采集完成: {len(all_landmarks)} 帧")
    
    if not all_landmarks:
        print("警告: 未采集到任何数据")
        return
    
    # 步骤 3: 数据处理
    print("\n[步骤 3] 处理运动数据...")
    
    # 转换为 numpy 数组
    landmarks_array = [capturer.landmarks_to_array(lm) for lm in all_landmarks]
    
    # 平滑处理
    smoothed = processor.smooth_trajectory(landmarks_array)
    print(f"✓ 平滑处理完成，张量形状: {smoothed.shape}")
    
    # 获取设备信息
    device_info = processor.get_device_info()
    print(f"✓ GPU 信息: {device_info}")
    
    # 步骤 4: 动作映射
    print("\n[步骤 4] 将人类骨架映射到机器人关节...")
    
    joint_sequences = []
    for i in range(smoothed.shape[0]):
        frame_landmarks = smoothed[i]  # (33, 3)
        joint_config = mapper.map_landmarks_to_joints(frame_landmarks)
        joint_sequences.append(joint_config)
        
        if (i + 1) % 30 == 0:
            print(f"已映射 {i + 1} / {smoothed.shape[0]} 帧...")
    
    print(f"✓ 映射完成: {len(joint_sequences)} 个关节配置")

    # 步骤 4.5: 对关节角度做二次平滑（减少抖动）
    import torch as _torch
    angles_tensor = _torch.stack(joint_sequences)
    angles_tensor = MotionMapper.smooth_joint_angles(angles_tensor, window=8)
    joint_sequences = [angles_tensor[i] for i in range(angles_tensor.shape[0])]
    print(f"✓ 角度平滑完成")

    # 步骤 5: 输出结果
    print("\n[步骤 5] 保存结果...")
    
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    if args.output_joint_csv:
        csv_path = args.output_joint_csv
    else:
        csv_path = output_dir / 'joint_sequence.csv'
    
    mapper.export_to_csv(joint_sequences, str(csv_path))
    print(f"✓ 关节序列已保存: {csv_path}")
    
    # 导出第一帧的配置为 JSON
    json_path = output_dir / 'joint_config_sample.json'
    mapper.export_to_urdf(joint_sequences[0], str(json_path))
    print(f"✓ 关节配置示例已保存: {json_path}")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    
    # 清理资源
    capturer.close()


if __name__ == '__main__':
    main()
