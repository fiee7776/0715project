"""
视觉捕捉模块 (Motion Capture Module)
负责调用 MediaPipe 实时提取骨架关键点
支持摄像头输入和视频文件输入
"""

import mediapipe as mp
import cv2
import numpy as np
from typing import Optional, Tuple


class MotionCapturer:
    """
    运动捕捉器 - 利用 MediaPipe 实时识别人体关键点
    
    Attributes:
        model_complexity (int): 模型复杂度 (0, 1, 2)，0最轻量，2最准确
        static_image_mode (bool): 是否使用静态图像模式
    """
    
    def __init__(self, model_complexity: int = 1, static_image_mode: bool = False):
        """
        初始化 MediaPipe Pose 检测器
        
        Args:
            model_complexity: 模型复杂度，建议实时场景用1，精度优先用2
            static_image_mode: 静态模式适合单帧图片，动态视频用False
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True  # 启用平滑以减少抖动
        )
        self.frame_count = 0
    
    def get_landmarks_from_frame(self, frame: np.ndarray):
        """
        从单帧图像提取人体关键点
        
        Args:
            frame: OpenCV 读入的图像 (BGR格式)
            
        Returns:
            MediaPipe 关键点结果对象，包含33个关键点的三维坐标
        """
        # 将 BGR 转为 RGB（MediaPipe 需要 RGB 格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        self.frame_count += 1
        return results.pose_landmarks
    
    def draw_landmarks_on_frame(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """
        在视频帧上绘制骨架和关键点（用于可视化）
        
        Args:
            frame: 原始图像帧
            landmarks: 提取的关键点
            
        Returns:
            标注了骨架的图像帧
        """
        if landmarks is None:
            return frame
        
        # 使用 MediaPipe 的绘制工具
        mp_drawing = mp.solutions.drawing_utils
        annotated_frame = frame.copy()
        
        mp_drawing.draw_landmarks(
            annotated_frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        return annotated_frame
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> list:
        """
        处理视频文件，提取每一帧的关键点
        
        Args:
            video_path: 输入视频文件路径
            output_path: 可选的输出视频文件路径（带骨架标注）
            
        Returns:
            每帧的关键点数据列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        all_landmarks = []
        
        # 初始化输出视频写入器（如果需要）
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.get_landmarks_from_frame(frame)
            all_landmarks.append(landmarks)
            
            # 绘制并保存标注后的视频
            if out:
                annotated = self.draw_landmarks_on_frame(frame, landmarks)
                out.write(annotated)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"已处理 {frame_idx} 帧...")
        
        cap.release()
        if out:
            out.release()
            print(f"标注视频已保存: {output_path}")
        
        return all_landmarks
    
    def landmarks_to_array(self, landmarks) -> np.ndarray:
        """
        将 MediaPipe 关键点转为 numpy 数组便于后续处理
        
        Args:
            landmarks: MediaPipe PoseLandmarks 对象
            
        Returns:
            形状为 (33, 3) 的 numpy 数组，每行是 (x, y, z) 坐标
        """
        if landmarks is None:
            return np.zeros((33, 3))
        
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    def close(self):
        """释放 MediaPipe 资源"""
        self.pose.close()
