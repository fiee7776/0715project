# G1 铁山靠 —— 基于强化学习的人形机器人戏曲武打动作复现

> 从人体视频动作采集，到仿真训练，再到真机部署的完整 Sim2Real 链路实践。以宇树 G1 人形机器人复现中国戏曲经典亮相动作"铁山靠"。

---

## 项目简介

本项目基于 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)（NVIDIA IsaacLab 强化学习框架）与宇树 G1-29dof 人形机器人，探索"人体动作 → 机器人动作"的完整技术链路：通过视频提取人体运动数据，重定向到机器人骨架，在仿真环境中用强化学习训练出可执行该动作的控制策略，先在仿真中验证（Sim2Sim），再评估向真机部署（Sim2Real）的可行性。

本次复现目标动作为**"铁山靠"**——中国传统戏曲（京剧）中武将亮相的经典身段，动作幅度大、姿态张力强，对机器人全身协调控制和动态平衡能力是一次很好的检验。

## 技术背景

项目思路来自对 2025 年央视春晚人形机器人表演的技术拆解：机器人如何通过"动作捕捉 → 动作重定向 → 仿真强化学习 → 真机部署"这条链路学会人类的复杂动作。完整技术原理与背景介绍见 [`docs/项目介绍.pptx`](./docs/)。

## 技术链路

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────────┐     ┌─────────────┐
│  视频素材    │ --> │  人体动作提取 │ --> │  动作重定向     │ --> │  仿真强化学习    │ --> │  Sim2Sim /   │
│ (铁山靠动作) │     │   (GVHMR)    │     │    (GMR)       │     │ (IsaacLab+AMP)  │     │  Sim2Real    │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────────┘     └─────────────┘
```

| 阶段 | 工具/框架 | 说明 |
|---|---|---|
| 动作采集 | 自拍/网络视频素材 | 单人入镜、正面或斜前方拍摄、全身可见 |
| 人体动作提取 | [GVHMR](https://github.com/zju3dv/GVHMR) | 单目视频 → 人体 3D 动作序列（SMPL 格式），基于视觉深度学习模型逐帧估计关节点与全身姿态 |
| 动作重定向 | [GMR](https://github.com/YanjieZe/GMR)（General Motion Retargeting） | 将 SMPL 人体动作映射到 G1 机器人关节空间，基于逆运动学优化，兼顾末端姿态、关节位置/速度约束 |
| 仿真训练 | [IsaacLab](https://github.com/isaac-sim/IsaacLab) + [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | 基于 AMP（Adversarial Motion Priors）模仿学习，训练策略网络输出关节动作，使机器人在物理仿真中复现目标动作 |
| Sim2Sim | [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | 训练策略从 IsaacSim 迁移到 Mujoco 独立仿真器，验证策略是否过拟合于单一仿真器的物理特性 |
| Sim2Real | Unitree SDK2 + C++ 控制程序 | 部署到 G1 实体机器人（视条件推进，为项目后续阶段） |

## 仓库结构

```
.
├── docs/                      # 项目介绍 PPT、技术说明文档
├── motion_capture/             # 视频素材、GVHMR 提取结果
│   ├── raw_video/              # 原始拍摄视频
│   └── smpl_output/            # GVHMR 输出的人体动作序列
├── retargeting/                 # GMR 重定向脚本与结果
│   └── g1_tieshankao.csv/npz    # 重定向后的 G1 关节角度序列
├── unitree_rl_lab/               # 训练框架（子模块或独立部署）
├── deploy/                       # Sim2Sim / Sim2Real 部署代码（C++ 控制程序）
├── outputs/                      # 训练日志、checkpoint、导出策略（policy.onnx）
└── README.md
```

## 环境部署

### 服务器配置

本项目训练环境部署于 [算力自由（GPUFree.cn）](https://www.gpufree.cn/) 云 GPU 平台：

| 项目 | 配置 |
|---|---|
| 镜像 | 具身机器人 / IsaacSim 5.0 + IsaacLab 2.2.1 |
| GPU | RTX 4090 24GB |
| CPU / 内存 | 14 核 / 50GB |

详细环境搭建与训练过程记录见 [`docs/部署与训练工作流程.md`](./docs/)。

### 快速开始

```bash
# 1. 克隆 unitree_rl_lab 并安装
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
cd unitree_rl_lab && conda activate isaaclab && ./unitree_rl_lab.sh -i

# 2. 下载机器人 USD 模型文件
hf download unitreerobotics/unitree_model --repo-type dataset --local-dir ./unitree_model

# 3. 配置模型路径（编辑 source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py）
# UNITREE_MODEL_DIR = "<你的路径>/unitree_model"

# 4. 查看可用任务
./unitree_rl_lab.sh -l
```

## 动作数据准备（铁山靠）

1. 拍摄铁山靠动作参考视频（单人、正面/斜45°、全身入镜、背景干净、贴身衣物、关键姿态可停顿）
2. 用 GVHMR 提取人体 3D 动作序列
3. 用 GMR 将动作重定向到 G1-29dof 骨架，导出为训练可用的关节角度序列
4. 参考仓库内现有的 `Unitree-G1-29dof-Mimic-Dance-102` / `Mimic-Gangnanm-Style` 任务配置，新建"铁山靠"专属 Mimic 训练任务，接入重定向后的动作数据

## 训练

```bash
tmux new -s train
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-TieShanKao --num_envs 4096
```

训练过程使用 `tmux` 保持后台运行；checkpoint 定期自动保存，支持从最近存档续训（`--resume --load_run <时间戳> --checkpoint <model_xxx.pt>`）。

## 推理与可视化

```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Mimic-TieShanKao \
    --load_run <运行时间戳> --checkpoint <model_xxx.pt>
```

推理时会自动导出部署用的 `policy.onnx` / `policy.pt`，路径位于对应运行目录下的 `exported/`。

## Sim2Sim 部署

将导出的 `policy.onnx` 替换至：
```
deploy/robots/g1_29dof/config/policy/mimic/tieshankao/exported/policy.onnx
```

编译并运行控制程序：
```bash
cd deploy/robots/g1_29dof
mkdir build && cd build
cmake .. && make
./g1_ctrl --network lo   # 本地 Mujoco 仿真联调
```

配合 [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) 启动仿真场景，验证策略在独立仿真器中的表现。

## 项目进展

- [x] 服务器环境搭建（GPUFree.cn + IsaacLab）
- [x] unitree_rl_lab 安装与验证（Velocity 任务训练跑通）
- [x] 训练结果 Sim2Sim 移交测试
- [ ] 铁山靠动作视频拍摄
- [ ] GVHMR 人体动作提取
- [ ] GMR 动作重定向到 G1
- [ ] 铁山靠 Mimic 任务训练
- [ ] 训练结果 Sim2Sim 验证
- [ ] Sim2Real 真机部署（视条件推进）

## 致谢

本项目基于以下开源项目构建：

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)：仿真与训练基础框架
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)：宇树机器人强化学习训练与部署框架
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)：Sim2Sim 仿真验证
- [GVHMR](https://github.com/zju3dv/GVHMR)：单目视频人体动作提取
- [GMR](https://github.com/YanjieZe/GMR)：跨形态动作重定向

## License

本项目遵循 [Apache-2.0 License](./LICENSE)，与上游 `unitree_rl_lab` 保持一致。
