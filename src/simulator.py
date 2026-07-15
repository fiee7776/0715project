"""
机器人仿真模块 (Robot Simulator Module)
利用 PyBullet 物理引擎进行虚拟机器人仿真
支持加载 CSV 关节数据并播放机器人动作
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
from pathlib import Path
from typing import List, Optional
import pandas as pd


class RobotSimulator:
    """
    机器人仿真器 - 在虚拟环境中控制机器人动作
    
    Attributes:
        client_id (int): PyBullet 客户端 ID
        robot_id (int): 加载的机器人模型 ID
        joint_ids (list): 控制关节的索引列表
    """
    
    def __init__(self, urdf_path: str, use_gui: bool = True, gravity: float = -9.8):
        """
        初始化仿真环境
        
        Args:
            urdf_path: 机器人 URDF 模型文件路径
            use_gui: 是否使用图形界面（True=可视化，False=仅计算）
            gravity: 重力加速度
        """
        # 连接 PyBullet
        if use_gui:
            self.client_id = p.connect(p.GUI)
            print("[RobotSimulator] 已连接 PyBullet GUI 模式")
        else:
            self.client_id = p.connect(p.DIRECT)
            print("[RobotSimulator] 已连接 PyBullet 直接模式")
        
        # 设置物理参数
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, gravity)
        p.setPhysicsEngineParameter(
            numSubSteps=5,
            numSolverIterations=100
        )
        
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        
        # 加载机器人模型
        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")
        
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 1.0],  # 机器人从离地 1 米处开始
            useFixedBase=True  # 固定基座，机器人不会因为手臂运动而飘动
        )
        
        print(f"[RobotSimulator] 已加载机器人模型: {urdf_path}")
        
        # 获取所有活跃关节
        self._init_joints()
        
        # 设置腿部关节为固定位置（稳定机器人）
        self._fix_leg_joints()
        
        # 设置渲染参数
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        self.simulation_time = 0.0
    
    def _init_joints(self):
        """初始化关节信息"""
        num_joints = p.getNumJoints(self.robot_id)
        
        # 按 URDF 定义顺序排列，与 CSV 列顺序一致
        control_joints = [
            'torso_bend', 'torso_twist',
            'left_shoulder', 'left_elbow',
            'right_shoulder', 'right_elbow',
            'left_hip', 'left_knee',
            'right_hip', 'right_knee',
        ]
        self.joint_ids = []
        self.joint_names = []
        
        print(f"[RobotSimulator] 机器人关节数: {num_joints}")
        print(f"[RobotSimulator] 寻找控制关节: {control_joints}")
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # 如果这个关节是我们要控制的动作关节
            if joint_name in control_joints and joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_ids.append(i)
                self.joint_names.append(joint_name)
                print(f"  [{i}] {joint_name} - 可控制")
    
    def _fix_leg_joints(self):
        """
        固定腿部关节，使机器人稳定站立
        """
        # 现在肩膀和髋部也参与控制，不再做额外锁定。
        return
    
    def set_joint_angles(self, angles: np.ndarray, max_force: float = 800.0):
        """
        设置关节角度

        Args:
            angles: 关节角度数组（度），映射模块已转换到机器人坐标系
            max_force: 最大关节力矩
        """
        if angles is None or len(angles) == 0:
            return

        angles = np.asarray(angles, dtype=float).copy()
        angles = np.deg2rad(angles)  # CSV 存储的是度，URDF 用弧度

        num = min(len(angles), len(self.joint_ids))
        for i in range(num):
            try:
                p.setJointMotorControl2(
                    self.robot_id,
                    self.joint_ids[i],
                    p.POSITION_CONTROL,
                    targetPosition=float(angles[i]),
                    targetVelocity=0.0,
                    force=max_force,
                    positionGain=0.4,
                    velocityGain=0.8,
                )
            except Exception:
                pass
    
    def step_simulation(self, time_step: float = 1.0 / 240.0):
        """
        执行一步仿真
        
        Args:
            time_step: 仿真时间步长
        """
        if not p.isConnected(self.client_id):
            raise RuntimeError("Not connected to physics server.")

        p.stepSimulation()
        self.simulation_time += time_step
    
    def play_joint_sequence(self, joint_sequence: List[np.ndarray], 
                           frame_rate: float = 30.0, 
                           loop: bool = False,
                           max_frames: Optional[int] = None):
        """
        播放关节序列动作
        
        Args:
            joint_sequence: 关节序列列表 [(4,), (4,), ...]
            frame_rate: 视频帧率（用于计算时间）
            loop: 是否循环播放
            max_frames: 最多播放多少帧（None=全部）
        """
        time_step = 1.0 / frame_rate
        total_frames = len(joint_sequence)
        
        if max_frames:
            total_frames = min(max_frames, total_frames)
        
        print(f"\n[RobotSimulator] 开始播放动作序列")
        print(f"  总帧数: {total_frames}")
        print(f"  帧率: {frame_rate} FPS")
        print(f"  循环: {'是' if loop else '否'}")
        print(f"  预计时长: {total_frames / frame_rate:.1f} 秒\n")
        
        frame_idx = 0
        loop_count = 0
        
        try:
            while True:
                if frame_idx >= total_frames:
                    if not loop:
                        print(f"✓ 播放完成 ({total_frames} 帧)")
                        break
                    else:
                        loop_count += 1
                        frame_idx = 0
                        print(f"↻ 循环播放 (第 {loop_count} 次)")
                
                # 获取当前帧的关节角度
                current_angles = joint_sequence[frame_idx]
                
                # 设置关节角度
                self.set_joint_angles(current_angles)
                
                # 执行多步仿真以保证稳定性
                for _ in range(4):
                    self.step_simulation(time_step / 4.0)
                
                # 打印进度
                if (frame_idx + 1) % 30 == 0:
                    elapsed = (frame_idx + 1) / frame_rate
                    print(f"已播放: {frame_idx + 1}/{total_frames} 帧 ({elapsed:.1f}s)")
                
                frame_idx += 1
                
                # 控制播放速度（小延迟即可，避免吃满 CPU）
                time.sleep(0.002)
        
        except KeyboardInterrupt:
            print(f"\n⏸ 播放已暂停 (第 {frame_idx} 帧)")
        except p.error as exc:
            if "Not connected to physics server" in str(exc):
                print(f"\n⏹ 检测到 PyBullet 已断开连接，停止播放 (第 {frame_idx} 帧)")
            else:
                raise
    
    def get_robot_state(self) -> dict:
        """
        获取机器人当前状态
        
        Returns:
            包含位置、方向、速度等信息的字典
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        joint_states = p.getJointStates(self.robot_id, self.joint_ids)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        return {
            'base_position': np.array(pos),
            'base_orientation': np.array(orn),
            'base_linear_velocity': np.array(lin_vel),
            'base_angular_velocity': np.array(ang_vel),
            'joint_positions': np.array(joint_positions),
            'joint_velocities': np.array(joint_velocities),
            'joint_names': self.joint_names,
            'simulation_time': self.simulation_time
        }
    
    def reset_robot(self, base_position: Optional[list] = None):
        """
        重置机器人到初始位置
        
        Args:
            base_position: 新的基座位置，默认为 (0, 0, 1.0)
        """
        if base_position is None:
            base_position = [0, 0, 1.0]
        
        p.resetBasePositionAndOrientation(
            self.robot_id,
            base_position,
            [0, 0, 0, 1]
        )
        
        # 重置所有关节到 0
        for joint_id in self.joint_ids:
            p.resetJointState(self.robot_id, joint_id, 0.0)
        
        self.simulation_time = 0.0
        print("[RobotSimulator] 机器人已重置")
    
    def load_joint_data_from_csv(self, csv_path: str) -> List[np.ndarray]:
        """
        从 CSV 文件加载关节数据
        
        Args:
            csv_path: CSV 文件路径
            
        Returns:
            关节序列列表
        """
        df = pd.read_csv(csv_path)
        
        # 处理列表形式的字符串数据 "[value]" -> value
        def parse_value(val):
            if isinstance(val, str):
                # 移除括号和空格
                val = val.strip('[]').strip()
            try:
                return float(val)
            except (ValueError, TypeError):
                return np.nan
        
        # 优先使用语义化列名，兼容旧版
        preferred_cols_10 = [
            'torso_bend', 'torso_twist',
            'left_shoulder', 'left_elbow',
            'right_shoulder', 'right_elbow',
            'left_hip', 'left_knee',
            'right_hip', 'right_knee',
        ]
        preferred_cols_8 = [
            'left_shoulder', 'left_elbow',
            'right_shoulder', 'right_elbow',
            'left_hip', 'left_knee',
            'right_hip', 'right_knee',
        ]
        
        if all(col in df.columns for col in preferred_cols_10):
            joint_cols = preferred_cols_10
        elif all(col in df.columns for col in preferred_cols_8):
            joint_cols = preferred_cols_8
        else:
            joint_cols = [col for col in df.columns if col.startswith('joint_')]
        
        processed_data = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            angles = [parse_value(row[col]) for col in joint_cols]
            
            # 如果是旧格式（8列），则在前面插入两列躯干数据
            if len(joint_cols) == 8 and not any(np.isnan(angles)):
                torso_bend = 0.0
                torso_twist = 0.0
                angles = [torso_bend, torso_twist] + angles
            elif len(joint_cols) == 4 and not any(np.isnan(angles)):
                # 兼容更旧的格式（只有肘、膝）
                # 从关节角度推断躯干运动：当手臂和腿向前摆时，躯干可能在摆
                left_elbow = float(angles[0])
                right_elbow = float(angles[1])
                left_knee = float(angles[2])
                right_knee = float(angles[3])
                
                # 躯干弯曲：当肘打开（角度变小）时，躯干向前倾
                avg_arm_open = (180 - left_elbow + 180 - right_elbow) / 2.0
                torso_bend = np.clip((avg_arm_open - 90) * 0.3, -40, 40)  # 缩放因子 0.3
                
                # 躯干旋转：左右手臂交替运动时躯干旋转
                arm_diff = (left_elbow - right_elbow) * 0.15  # 缩放因子 0.15
                torso_twist = np.clip(arm_diff, -70, 70)
                
                left_shoulder = float(np.clip(left_elbow * 0.75, 0.0, 180.0))
                right_shoulder = float(np.clip(right_elbow * 0.75, 0.0, 180.0))
                left_hip = float(np.clip(left_knee * 0.85, 0.0, 180.0))
                right_hip = float(np.clip(right_knee * 0.85, 0.0, 180.0))
                
                angles = [
                    torso_bend, torso_twist,
                    left_shoulder, left_elbow,
                    right_shoulder, right_elbow,
                    left_hip, left_knee,
                    right_hip, right_knee,
                ]
            
            if not any(np.isnan(angles)):
                processed_data.append(np.array(angles))
        
        joint_data = np.array(processed_data)
        
        print(f"[RobotSimulator] 已加载 CSV 数据")
        print(f"  路径: {csv_path}")
        print(f"  帧数: {len(joint_data)}")
        print(f"  关节数: {joint_data.shape[1] if joint_data.shape else 0}")
        
        # 转换为 numpy 数组列表
        sequence = [joint_data[i] for i in range(len(joint_data))]
        
        return sequence
    
    def set_camera(self, distance: float = 2.0, yaw: float = 45, pitch: float = -30):
        """
        调整摄像机视角
        
        Args:
            distance: 摄像机距离目标的距离
            yaw: 绕 Z 轴旋转角（度）
            pitch: 俯仰角（度）
        """
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=[0, 0, 1.0]
        )
    
    def close(self):
        """关闭仿真"""
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
            print("[RobotSimulator] 仿真已关闭")
    
    def __enter__(self):
        """支持 with 语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.close()
