# G1 铁山靠：基于强化学习的人形机器人动作复现

本项目面向宇树 G1-29dof 人形机器人，记录从单目视频动作提取、跨形态动作重定向、IsaacLab 强化学习训练，到 Sim2Sim 部署验证的完整流程。当前实验任务在代码中命名为 `Input5`，对应的训练环境 ID 为 `Unitree-G1-29dof-Mimic-Input5`。

## 技术链路

```text
视频素材
  -> GVHMR 提取人体三维动作
  -> GMR 重定向到 G1-29dof
  -> IsaacLab + unitree_rl_lab 模仿学习与强化学习
  -> 导出策略
  -> Mujoco Sim2Sim 验证
  -> Sim2Real（后续）
```

| 阶段 | 工具或框架 | 说明 |
|---|---|---|
| 人体动作提取 | [GVHMR](https://github.com/zju3dv/GVHMR) | 将单目视频转换为 SMPL 人体三维动作序列 |
| 动作重定向 | [GMR](https://github.com/YanjieZe/GMR) | 将人体动作映射到 G1 关节空间 |
| 仿真训练 | [IsaacLab](https://github.com/isaac-sim/IsaacLab) + [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | 训练机器人跟踪目标动作并保持动态稳定 |
| Sim2Sim | [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | 在独立物理仿真器中验证策略迁移效果 |
| Sim2Real | Unitree SDK2 | 后续部署到 G1 实体机器人 |

## 仓库内容

本仓库不重复保存完整的 GVHMR、GMR、IsaacLab 和 unitree_rl_lab 上游源码，只保存本项目新增或修改的代码、配置、动作数据和部署参数。

```text
.
├── configs/                         # Stage1、Stage2a、Stage2b 和微调配置
├── deploy/
│   └── configs/                     # 部署参数
├── docs/
│   └── images/                      # 项目展示图片
├── motion_capture/
│   └── gvhmr_overrides/             # GVHMR 环境适配文件
├── outputs/
│   └── README.md                    # 大模型和训练产物存放说明
├── patches/                         # 训练流程补丁
├── retargeting/
│   ├── data/                        # G1 动作 CSV
│   └── gmr_overrides/               # GMR 修改代码
├── scripts/                         # 数据处理、检查和打包脚本
└── training/
    └── unitree_overrides/           # unitree_rl_lab 修改代码
```

## 环境

项目训练环境：

| 项目 | 配置 |
|---|---|
| 云平台 | GPUFree |
| 系统 | Ubuntu 22.04 |
| GPU | RTX 4090 24GB |
| CPU / 内存 | 14 核 / 50GB |
| 仿真环境 | IsaacSim 5.0 + IsaacLab 2.2.1 |

上游依赖：

```bash
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
git clone https://github.com/YanjieZe/GMR.git
git clone https://github.com/zju3dv/GVHMR.git
```

依赖版本与安装方式以各上游项目文档为准。机器人模型可从 Unitree 官方模型仓库获取。

## 应用本仓库修改

将 unitree_rl_lab 修改文件按原路径覆盖到上游仓库：

```bash
cp -a training/unitree_overrides/source/. /path/to/unitree_rl_lab/source/
```

将 GMR 修改文件按原路径覆盖到上游仓库：

```bash
cp -a retargeting/gmr_overrides/. /path/to/GMR/
```

`training/unitree_overrides/` 顶层还保留了部分关键文件的独立快照，便于比较不同训练阶段。

## 动作数据

`retargeting/data/input5_v2_50hz.csv` 是当前纳入版本控制的 G1 动作数据。训练用二进制数据、原始视频和完整 GVHMR 输出不进入普通 Git 历史。

主要处理脚本位于 `scripts/`：

- `make_input5_v2.py`：生成 Input5 处理版本。
- `analyze_tensorboard_run.py`：分析训练过程。
- `reset_checkpoint_noise.py`：调整检查点中的探索参数。
- `build_input5_windows_bundle.py`：构建部署测试包。
- `run_input5_mujoco.py`：运行 Mujoco 验证流程。

## 训练

基础训练任务：

```bash
cd /path/to/unitree_rl_lab
tmux new -s input5-train
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-Input5 --num_envs 4096
```

低探索微调任务：

```bash
./unitree_rl_lab.sh -t \
  --task Unitree-G1-29dof-Mimic-Input5-Finetune \
  --resume \
  --load_run <运行目录> \
  --checkpoint <model_xxx.pt>
```

推理与可视化：

```bash
./unitree_rl_lab.sh -p \
  --task Unitree-G1-29dof-Mimic-Input5 \
  --load_run <运行目录> \
  --checkpoint <model_xxx.pt>
```

Stage1、Stage2a、Stage2b 的关键配置快照保存在 `configs/`，用于记录训练策略的演进过程。

## 模型与训练产物

以下文件不直接提交到普通 Git 历史：

- PyTorch 模型和检查点：`.pt`、`.pth`、`.ckpt`
- 导出策略：`.onnx`
- 二进制动作数据：`.npz`、`.pkl`
- 原始视频、完整日志、TensorBoard 数据和压缩包
- Conda 环境、依赖缓存和临时文件

最终策略、检查点和演示视频应通过 GitHub Releases 或外部存储发布，仓库中保留下载说明和校验值。

## 当前进展

- [x] GPUFree + IsaacLab 环境搭建
- [x] unitree_rl_lab 安装与基础任务验证
- [x] GVHMR 动作提取流程验证
- [x] GMR 到 G1-29dof 动作重定向
- [x] Input5 50Hz 动作 CSV 生成
- [x] Input5 Mimic 任务注册
- [x] Stage1、Stage2a、Stage2b 配置整理
- [x] 低探索微调配置
- [x] Mujoco 运行脚本与部署参数整理
- [ ] 完整 Sim2Sim 结果验收
- [ ] Sim2Real 真机部署

## 上游项目

- [IsaacLab](https://github.com/isaac-sim/IsaacLab)
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [GVHMR](https://github.com/zju3dv/GVHMR)
- [GMR](https://github.com/YanjieZe/GMR)

使用本仓库中的上游适配代码时，请同时遵守对应上游项目的许可证。项目自有内容的统一许可证以仓库后续补充的 `LICENSE` 文件为准。
