"""
动作映射模块 (Motion Mapping Module)
将人类骨架关键点映射到机器人关节配置空间

坐标系说明:
  MediaPipe: X=右, Y=下, Z=进入屏幕（相机坐标系）
  URDF/PyBullet: X=前, Y=左, Z=上（机器人坐标系）
  转换公式: robot_x = -mp_z, robot_y = -mp_x, robot_z = -mp_y
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import json


class MotionMapper:
    """
    运动映射器 - 将 MediaPipe 33 个关键点映射为 10 个机器人关节角度（度）。
    输出已转换到 URDF 关节轴约定，simulator 只需 deg→rad。
    """

    def __init__(self, robot_config: Optional[Dict] = None):
        self.robot_config = robot_config or self._default_robot_config()
        self.mapping_table = self._build_mapping_table()

    @staticmethod
    def _default_robot_config() -> Dict:
        return {
            'robot_type': 'humanoid',
            'dof': 10,
            'joint_limits': {
                'torso_bend':     {'min': -40, 'max': 40},
                'torso_twist':    {'min': -70, 'max': 70},
                'left_shoulder':  {'min': -90, 'max': 90},
                'right_shoulder': {'min': -90, 'max': 90},
                'left_elbow':     {'min': 0, 'max': 150},
                'right_elbow':    {'min': 0, 'max': 150},
                'left_hip':       {'min': -90, 'max': 90},
                'right_hip':      {'min': -90, 'max': 90},
                'left_knee':      {'min': 0, 'max': 150},
                'right_knee':     {'min': 0, 'max': 150},
            }
        }

    @staticmethod
    def _joint_order() -> List[str]:
        """关节顺序 = URDF 定义顺序 = CSV 列顺序。"""
        return [
            'torso_bend', 'torso_twist',
            'left_shoulder', 'left_elbow',
            'right_shoulder', 'right_elbow',
            'left_hip', 'left_knee',
            'right_hip', 'right_knee',
        ]

    def _build_mapping_table(self) -> Dict[str, int]:
        return {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13,    'right_elbow': 14,
            'left_wrist': 15,    'right_wrist': 16,
            'left_hip': 23,      'right_hip': 24,
            'left_knee': 25,     'right_knee': 26,
            'left_ankle': 27,    'right_ankle': 28,
        }

    # ------------------------------------------------------------------
    # 坐标系转换
    # ------------------------------------------------------------------
    @staticmethod
    def _to_robot_frame(landmarks: torch.Tensor) -> torch.Tensor:
        """
        将 MediaPipe 坐标系转换为 URDF 机器人坐标系。

        MediaPipe:  X=右, Y=下, Z=进屏幕（右手系）
        URDF:       X=前, Y=左, Z=上  （右手系）

        转换矩阵 (行列式=+1，纯旋转无镜像):
            robot_x =  mp_z
            robot_y = -mp_x
            robot_z = -mp_y
        """
        out = torch.zeros_like(landmarks)
        out[:, 0] = landmarks[:, 2]    # robot_x = mp_z
        out[:, 1] = -landmarks[:, 0]   # robot_y = -mp_x
        out[:, 2] = -landmarks[:, 1]   # robot_z = -mp_y
        return out

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def map_landmarks_to_joints(self, landmarks: torch.Tensor) -> torch.Tensor:
        """(33,3) MediaPipe → (10,) 关节角度（度，机器人坐标系）。"""
        device = landmarks.device
        # 先转换到机器人坐标系
        robot_lm = self._to_robot_frame(landmarks)
        kp = {name: robot_lm[idx] for name, idx in self.mapping_table.items()
              if idx < robot_lm.shape[0]}
        angles = self._compute_joint_angles(kp)
        return self._apply_constraints(angles).to(device)

    # ------------------------------------------------------------------
    # 角度计算核心（全部使用 atan2，数值稳定）
    # 所有计算均在 URDF 机器人坐标系中进行
    # ------------------------------------------------------------------
    def _compute_joint_angles(self, kp: Dict[str, torch.Tensor]) -> torch.Tensor:
        dev = kp['left_shoulder'].device
        angles = []

        shoulder_mid = (kp['left_shoulder'] + kp['right_shoulder']) / 2
        hip_mid = (kp['left_hip'] + kp['right_hip']) / 2
        torso_vec = shoulder_mid - hip_mid   # hip→shoulder（机器人坐标系）

        # ---- ① torso_bend ----
        # URDF: torso_bend 绕 X 轴旋转（前倾=正，后仰=负）
        # 在机器人坐标系中: 用 (y, z) 平面计算倾斜角
        #   robot_z > 0 表示肩在髋之上（站立）
        #   robot_y != 0 表示左右倾斜
        bend = torch.rad2deg(self._safe_atan2(torso_vec[1], torso_vec[2]))
        angles.append(torch.clamp(bend, -40.0, 40.0))

        # ---- ② torso_twist ----
        # URDF: torso_twist 绕 Z 轴旋转（左转=正，右转=负）
        # 在机器人坐标系中: 用 (x, y) 平面计算偏航角
        s_line = kp['right_shoulder'] - kp['left_shoulder']
        h_line = kp['right_hip'] - kp['left_hip']
        s_yaw = self._safe_atan2(s_line[1], s_line[0])  # (y, x) in robot frame
        h_yaw = self._safe_atan2(h_line[1], h_line[0])
        twist = torch.rad2deg(s_yaw - h_yaw)
        angles.append(torch.clamp(twist, -70.0, 70.0))

        # ---- ③⑤ shoulder abduction ----
        # URDF: 肩关节绕 Y 轴旋转
        #   +Y旋转 → 手臂向后/向下,  -Y旋转 → 手臂向前/向上
        #   计算 cross(torso_vec, upper_arm) 的 Y 分量确定方向
        angles.append(self._shoulder_angle(torso_vec, kp['left_shoulder'], kp['left_elbow']))
        angles.append(self._shoulder_angle(torso_vec, kp['right_shoulder'], kp['right_elbow']))

        # ---- ④⑥ elbow flexion ----
        # URDF: 肘关节绕 Y 轴旋转 (0~180°, 直=0, 弯=180)
        angles.append(self._limb_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist']))
        angles.append(self._limb_angle(kp['right_shoulder'], kp['right_elbow'], kp['right_wrist']))

        # ---- ⑦⑨ hip flexion ----
        # URDF: 髋关节绕 Y 轴旋转
        #   +Y旋转 → 腿向后,  -Y旋转 → 腿向前
        angles.append(self._hip_angle(torso_vec, kp['left_hip'], kp['left_knee']))
        angles.append(self._hip_angle(torso_vec, kp['right_hip'], kp['right_knee']))

        # ---- ⑧⑩ knee flexion ----
        # URDF: 膝关节绕 Y 轴旋转 (0~180°, 直=0, 弯=180)
        angles.append(self._limb_angle(kp['left_hip'], kp['left_knee'], kp['left_ankle']))
        angles.append(self._limb_angle(kp['right_hip'], kp['right_knee'], kp['right_ankle']))

        return torch.stack([a.squeeze() for a in angles]).to(dev)

    # ------------------------------------------------------------------
    # 稳定的角度辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_atan2(y, x):
        """atan2 with epsilon to avoid NaN on (0,0)."""
        return torch.atan2(y, x + 1e-8)

    @staticmethod
    def _vec_angle_atan2(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        两向量夹角（度，0~180），用 atan2(|cross|, dot) 计算，数值稳定。
        """
        c = torch.cross(v1, v2, dim=-1)
        cross_mag = torch.norm(c)
        dot = torch.dot(v1, v2)
        angle = torch.atan2(cross_mag, dot)   # [0, pi]
        return torch.rad2deg(angle)

    @staticmethod
    def _shoulder_angle(torso_vec: torch.Tensor,
                        shoulder: torch.Tensor,
                        elbow: torch.Tensor) -> torch.Tensor:
        """
        肩外展角 → robot: arm-down=-90, horizontal=0, arm-up=+90

        URDF 肩关节绕 Y 轴旋转:
          正Y旋转 → 手臂向下/向后,  负Y旋转 → 手臂向上/向前

        使用 cross(torso_vec, upper_arm) 的 Y 分量确定正负号:
          Y<0 → 手臂在躯干前方 → 负角度（手臂抬起）
          Y>0 → 手臂在躯干后方 → 正角度（手臂后摆）
        """
        upper_arm = elbow - shoulder
        theta = MotionMapper._vec_angle_atan2(torso_vec, upper_arm)
        c = torch.cross(torso_vec, upper_arm, dim=-1)
        sign = -torch.sign(c[1] + 1e-8)  # 取反以匹配 URDF Y 轴旋转约定
        return sign * (90.0 - theta)

    @staticmethod
    def _hip_angle(torso_vec: torch.Tensor,
                   hip: torch.Tensor,
                   knee: torch.Tensor) -> torch.Tensor:
        """
        髋屈伸角 → robot: leg-down=-90, horizontal=0, leg-back=+90

        URDF 髋关节绕 Y 轴旋转:
          正Y旋转 → 腿向后,  负Y旋转 → 腿向前

        使用 cross(torso_vec, thigh) 的 Y 分量确定正负号:
          Y<0 → 腿在躯干前方 → 负角度（抬腿向前）
          Y>0 → 腿在躯干后方 → 正角度（腿向后摆）
        """
        thigh = knee - hip
        theta = MotionMapper._vec_angle_atan2(torso_vec, thigh)
        c = torch.cross(torso_vec, thigh, dim=-1)
        sign = -torch.sign(c[1] + 1e-8)  # 取反以匹配 URDF Y 轴旋转约定
        return sign * (90.0 - theta)

    @staticmethod
    def _limb_angle(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """
        肘/膝弯曲角 → robot: straight=0, fully-bent=180
        """
        theta = MotionMapper._vec_angle_atan2(p1 - p2, p3 - p2)
        return 180.0 - theta

    # ------------------------------------------------------------------
    # 后处理：角度平滑
    # ------------------------------------------------------------------
    @staticmethod
    def smooth_joint_angles(angles_sequence: 'torch.Tensor',
                            window: int = 15) -> 'torch.Tensor':
        """
        对关节角度序列做高斯平滑（减少抖动）。
        Args:
            angles_sequence: (T, 10) tensor
            window: 平滑窗口
        Returns:
            (T, 10) smoothed tensor
        """
        from scipy.ndimage import gaussian_filter1d
        data = angles_sequence.cpu().numpy()
        smoothed = gaussian_filter1d(data, sigma=window / 3.0, axis=0)
        return torch.from_numpy(smoothed).float().to(angles_sequence.device)

    # ------------------------------------------------------------------
    # 约束 & 导出
    # ------------------------------------------------------------------
    def _apply_constraints(self, angles: torch.Tensor) -> torch.Tensor:
        c = angles.clone()
        for idx, name in enumerate(self._joint_order()):
            if idx >= c.shape[0]:
                break
            lim = self.robot_config['joint_limits'].get(name)
            if lim:
                c[idx] = torch.clamp(c[idx], float(lim['min']), float(lim['max']))
        return c

    def export_to_csv(self, sequence: List[torch.Tensor], path: str) -> None:
        import csv
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(self._joint_order()[:sequence[0].shape[0]])
            for fr in sequence:
                w.writerow(fr.cpu().numpy().tolist())

    def export_to_urdf(self, config: torch.Tensor, path: str) -> None:
        with open(path, 'w') as f:
            json.dump({
                'robot_type': self.robot_config['robot_type'],
                'joints': config.cpu().numpy().tolist(),
                'timestamp': str(np.datetime64('now'))
            }, f, indent=2)

    def get_mapping_info(self) -> Dict:
        return {
            'robot_type': self.robot_config['robot_type'],
            'dof': self.robot_config['dof'],
            'mapped_joints': len(self.mapping_table),
            'mapping_table': self.mapping_table
        }
