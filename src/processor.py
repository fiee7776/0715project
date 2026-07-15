"""
数据处理模块 (Motion Processing Module)
利用 PyTorch 和 RTX 4070 GPU 加速进行坐标平滑与空间映射
"""

import torch
import numpy as np
from typing import Optional, List
import cv2


class MotionProcessor:
    """
    运动数据处理器 - 负责坐标平滑、归一化和空间变换
    
    Attributes:
        device (str): 计算设备 ('cuda' 或 'cpu')
        smoothing_window (int): 平滑窗口大小
    """
    
    def __init__(self, smoothing_window: int = 5, use_cuda: bool = True):
        """
        初始化处理器
        
        Args:
            smoothing_window: 高斯平滑的窗口大小
            use_cuda: 是否使用 GPU 加速
        """
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.smoothing_window = smoothing_window
        print(f"[MotionProcessor] 使用设备: {self.device}")
        
        if use_cuda and torch.cuda.is_available():
            print(f"GPU 信息: {torch.cuda.get_device_name(0)}")
            print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> torch.Tensor:
        """
        将关键点坐标归一化到 [-1, 1] 范围
        
        Args:
            landmarks: 形状为 (33, 3) 的 numpy 数组
            
        Returns:
            归一化后的 PyTorch 张量，已转移到 GPU
        """
        # 转换为 PyTorch 张量
        tensor = torch.from_numpy(landmarks).float().to(self.device)
        
        # 计算每个坐标的最小值和最大值
        min_vals = tensor.min(dim=0).values
        max_vals = tensor.max(dim=0).values
        
        # 避免除以零
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        # 归一化到 [-1, 1]
        normalized = 2 * (tensor - min_vals) / range_vals - 1
        return normalized
    
    def smooth_trajectory(self, landmarks_sequence: List[np.ndarray],
                         window_size: Optional[int] = None) -> torch.Tensor:
        """
        使用高斯滤波平滑关键点轨迹（减少噪声和抖动）

        Args:
            landmarks_sequence: 时间序列的关键点 [(33,3), (33,3), ...]
            window_size: 平滑窗口大小，默认使用初始化时的值

        Returns:
            平滑后的轨迹张量，形状为 (T, 33, 3)
        """
        if window_size is None:
            window_size = self.smoothing_window

        from scipy.ndimage import gaussian_filter1d

        # 直接在 CPU 上做高斯平滑，避免无意义的 GPU 往返
        sequence = np.stack(landmarks_sequence, axis=0)  # (T, 33, 3)
        sigma = window_size / 3.0
        smoothed = gaussian_filter1d(sequence, sigma=sigma, axis=0)

        return torch.from_numpy(smoothed).float().to(self.device)
    
    def compute_joint_angles(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        从骨架关键点计算关节角度（适用于机器人控制）
        
        利用三个点计算角度：parent -> joint -> child
        
        Args:
            landmarks: 形状为 (33, 3) 的关键点张量
            
        Returns:
            关节角度张量
        """
        # 定义主要关节链（简化版）
        # 格式: (parent_idx, joint_idx, child_idx)
        joint_chains = [
            (11, 13, 15),  # 左肩 -> 左肘 -> 左腕
            (12, 14, 16),  # 右肩 -> 右肘 -> 右腕
            (23, 25, 27),  # 左髋 -> 左膝 -> 左脚踝
            (24, 26, 28),  # 右髋 -> 右膝 -> 右脚踝
        ]
        
        angles = []
        for parent_idx, joint_idx, child_idx in joint_chains:
            # 获取三个点
            p1 = landmarks[parent_idx]
            p2 = landmarks[joint_idx]
            p3 = landmarks[child_idx]
            
            # 计算向量
            v1 = p1 - p2
            v2 = p3 - p2
            
            # 计算夹角
            cos_angle = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
            angle = torch.acos(torch.clamp(cos_angle, -1, 1))
            angles.append(angle)
        
        return torch.stack(angles)
    
    def center_and_scale(self, landmarks: torch.Tensor) -> tuple:
        """
        以髋部中心为原点，对关键点进行中心化和缩放
        
        Args:
            landmarks: 形状为 (33, 3) 的关键点张量
            
        Returns:
            (中心化的关键点, 缩放因子)
        """
        # 使用中间的髋部关键点作为中心
        hip_center = (landmarks[23] + landmarks[24]) / 2  # 左右髋的中点
        
        # 中心化
        centered = landmarks - hip_center
        
        # 计算缩放因子（以身体高度归一化）
        height = torch.norm(landmarks[0] - landmarks[28])  # 鼻尖到右脚踝的距离
        scale_factor = 1.0 / (height + 1e-6)
        
        return centered * scale_factor, scale_factor
    
    def batch_process(self, batch_landmarks: List[np.ndarray]) -> torch.Tensor:
        """
        批量处理多个关键点帧

        Args:
            batch_landmarks: 多个 (33, 3) 数组的列表

        Returns:
            处理后的批量张量，形状为 (B, 33, 3)
        """
        batch = np.stack(batch_landmarks, axis=0)  # (B, 33, 3)
        tensor = torch.from_numpy(batch).float().to(self.device)

        # 向量化归一化，避免逐帧 CPU↔GPU 往返
        min_vals = tensor.min(dim=1, keepdim=True).values
        max_vals = tensor.max(dim=1, keepdim=True).values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        normalized = 2 * (tensor - min_vals) / range_vals - 1

        return normalized.to(self.device)
    
    def get_device_info(self) -> dict:
        """获取当前计算设备信息"""
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
            info['device_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['current_memory_gb'] = torch.cuda.memory_allocated() / 1e9
        
        return info
