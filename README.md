# Humanoid Dance Robot

基于计算机视觉的人体动作捕捉 + PyBullet 人形机器人舞蹈仿真系统。

通过摄像头或视频文件捕捉人体舞蹈动作，经 MediaPipe 提取 33 个骨架关键点，利用 PyTorch GPU 加速处理后，映射为 10 个机器人关节角度，最终在 PyBullet 物理仿真环境中驱动人形机器人还原舞蹈。

## 效果展示

| 原视频 + MediaPipe 骨架 | PyBullet 机器人仿真 |
|:---:|:---:|
| ![frame0](outputs/visualization/comparison_0000.png) | ![frame316](outputs/visualization/comparison_0316.png) |
| ![frame633](outputs/visualization/comparison_0633.png) | ![frame949](outputs/visualization/comparison_0949.png) |

完整对比视频见 [`outputs/visualization/comparison_video.mp4`](outputs/visualization/comparison_video.mp4)

## 系统架构

```
┌─────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  视频/摄像头  │───>│  MediaPipe    │───>│  PyTorch     │───>│  关节角度     │───>│  PyBullet    │
│  (OpenCV)   │    │  33个关键点    │    │  GPU平滑处理  │    │  映射+约束    │    │  物理仿真    │
└─────────────┘    └───────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
    capture.py         capture.py         processor.py        mapping.py         simulator.py
```

### 数据流

```
视频帧 (BGR)
  ↓ MediaPipe Pose
33个关键点 (x, y, z) × T帧
  ↓ 高斯滤波平滑 (scipy)
平滑后关键点 (T, 33, 3)
  ↓ 向量夹角计算 + URDF坐标系变换
10个关节角度 (度) × T帧
  ↓ 关节约束 + 二次平滑
最终关节序列 → CSV
  ↓ deg→rad 转换
PyBullet 关节驱动
```

## 项目结构

```
humanoid_dance_project/
├── main.py                          # 主程序：视频→关节角度CSV
├── simulate.py                      # 仿真入口：CSV→PyBullet播放
├── visualize.py                     # 可视化：生成对比图和视频
├── test_joint_control.py            # 关节控制测试
├── view_robot_model.py              # URDF模型查看器
│
├── src/                             # 核心模块
│   ├── capture.py                   # MediaPipe骨架检测
│   ├── processor.py                 # PyTorch数据处理(GPU)
│   ├── mapping.py                   # 关键点→关节角度映射
│   └── simulator.py                 # PyBullet仿真控制
│
├── assets/                          # 资源文件
│   ├── humanoid_robot_mediapipe_v2.urdf  # 人形机器人模型(10关节)
│   ├── humanoid_robot.urdf              # 简化模型(4关节)
│   ├── humanoid_robot_v2.urdf           # 中等模型(8关节)
│   ├── humanoid_robot_mediapipe.urdf    # 胶囊体模型(8关节)
│   ├── trapezoid_torso.obj              # 躯干网格
│   └── video6330050713560816015.mp4     # 输入舞蹈视频
│
├── outputs/                         # 输出文件
│   ├── joint_sequence.csv           # 关节角度序列(1109帧×10关节)
│   ├── joint_config_sample.json     # 单帧配置示例
│   ├── annotated_video.mp4          # 骨架标注视频
│   └── visualization/               # 可视化结果
│       ├── contact_sheet.png        # 8帧对比合集
│       ├── comparison_*.png         # 单帧对比图
│       └── comparison_video.mp4     # 全程对比视频
│
└── requirements.txt                 # Python依赖
```

## 实现原理

### 1. 骨架检测 (`capture.py`)

使用 MediaPipe Pose 从视频帧中检测 33 个人体关键点（3D坐标），包括鼻、肩、肘、腕、髋、膝、踝等。每帧输出 `(33, 3)` 的坐标矩阵。

### 2. 数据处理 (`processor.py`)

- **高斯轨迹平滑**：对每个关键点的时间序列做 1D 高斯滤波，消除检测噪声
- **GPU 加速**：利用 PyTorch + CUDA 在 RTX 4070 上并行处理
- **二次角度平滑**：映射后的关节角度再做一轮平滑，消除映射引入的抖动

### 3. 关节映射 (`mapping.py`)

核心难点：MediaPipe 输出的是 3D 关键点坐标，而机器人需要的是关节旋转角度。两者坐标系和运动学定义完全不同。

**映射策略：**

| 机器人关节 | 计算方法 | 角度约定 |
|:---:|:---|:---|
| `torso_bend` | 肩髋中心连线偏离竖直方向的角度 | 前倾+, 后仰- |
| `torso_twist` | 肩线与髋线在水平面的夹角差 | 左转+, 右转- |
| `left/right_shoulder` | 上臂与躯干向量的夹角 | 手臂下垂=-90°, 水平=0°, 上举=+90° |
| `left/right_elbow` | 上臂与前臂的夹角 | 伸直=0°, 弯曲=180° |
| `left/right_hip` | 大腿与躯干向量的夹角 | 垂下=-90°, 水平=0°, 后摆=+90° |
| `left/right_knee` | 大腿与小腿的夹角 | 伸直=0°, 弯曲=180° |

**数值稳定性：** 使用 `atan2(|cross|, dot)` 替代 `acos(dot)` 计算向量夹角，避免浮点精度问题导致的抖动。

**URDF 坐标系变换：** MediaPipe 坐标系（Y向下、Z向屏幕内）与 PyBullet 坐标系（Z向上）不同，映射时需要根据 URDF 关节轴方向做相应变换。

### 4. 物理仿真 (`simulator.py`)

- 使用 PyBullet 加载 URDF 人形机器人模型
- 通过 PD 位置控制器驱动 10 个关节
- 关节角度从度转弧度后直接驱动（映射模块已完成坐标系变换）

### 5. URDF 机器人模型

`humanoid_robot_mediapipe_v2.urdf` 包含 10 个可动关节：

```
base_link
├── torso_bend (X轴旋转, ±40°)
│   └── torso_twist (Z轴旋转, ±70°)
│       └── torso_link (梯形躯干)
│           ├── neck → head (固定)
│           ├── left_shoulder (Y轴旋转) → left_elbow → left_hand
│           ├── right_shoulder (Y轴旋转) → right_elbow → right_hand
│           ├── left_hip (Y轴旋转) → left_knee → left_foot
│           └── right_hip (Y轴旋转) → right_knee → right_foot
```

## 快速开始

### 环境要求

- Python 3.10+ (推荐 conda `robot_env` 环境)
- NVIDIA GPU (CUDA) — 可选，CPU 也可运行
- 依赖包见 `requirements.txt`

### 运行步骤

```bash
# 1. 从视频提取关节角度
python main.py --mode video --video_path assets/video6330050713560816015.mp4

# 2. PyBullet 仿真播放
python simulate.py

# 3. 生成对比可视化
python visualize.py --mode both

# 也可以用摄像头实时捕捉
python main.py --mode camera
```

### 主要参数

```bash
# main.py
--mode camera|video        # 输入模式
--video_path <path>        # 视频文件路径
--model_complexity 0|1|2   # MediaPipe模型复杂度(0轻量/2精确)
--output_joint_csv <path>  # 输出CSV路径

# simulate.py
--csv_file <path>          # 关节数据CSV
--urdf_file <path>         # URDF模型路径
--frame_rate <fps>         # 播放帧率
--loop                     # 循环播放
--max_frames <n>           # 最大帧数

# visualize.py
--mode images|video|both   # 输出模式
--max_frames <n>           # 最大帧数
```

## 技术要点

1. **向量夹角的数值稳定性**：使用 `atan2(|cross|, dot)` 而非 `acos(dot)`，避免夹角接近 0° 或 180° 时的浮点抖动
2. **两级平滑**：先平滑关键点轨迹，再平滑关节角度，双重降噪
3. **URDF 坐标系对齐**：映射模块输出的角度已转换到 URDF 关节轴约定，simulator 只需 deg→rad
4. **关节约束**：每个关节角度限制在 URDF 定义的物理范围内，防止模型穿模

## 依赖

```
opencv-python    mediapipe    torch    numpy    scipy
pandas           pybullet     matplotlib
```

## License

MIT
