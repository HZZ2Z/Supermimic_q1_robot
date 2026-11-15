# Q1 SuperMimic Pole Weaving 环境搭建

本指南用于绕杆训练、评估和回放。基础环境为 Linux、Python 3.11、MuJoCo 与 CUDA 12.8 版 PyTorch，Conda 环境名保持为 `env_q1`。

## 安装

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate env_q1
python -m pip install --no-deps -e .
```

普通 Python 包可按网络情况选择 PyPI 镜像；PyTorch CUDA wheel 使用 `requirements.txt` 声明的官方索引。已有 `env_q1` 时可直接激活并检查依赖，无需重复创建环境。

### 随分支保存的依赖源码

`3rdparty/SuperMimic/` 已作为普通目录保存，包含本版奖励修改，不再使用 Git 子模块。缺失时请从本仓库的 `reward_footflip` 分支恢复，不能直接覆盖为上游版本。

`shared/supermimic/upstream.json` 保留上游来源与原始哈希；本地差异见 [LOCAL_CHANGES.md](../3rdparty/SuperMimic/LOCAL_CHANGES.md)。源任务历史、模型与数据的身份仍由 `pole_task/source_manifest.json` 管理。

## 检查安装

```bash
python -c "import sys, torch, mujoco, mujoco_warp, warp, q1_supermimic, pole_task, expert.policy; print('Python:', sys.version.split()[0]); print('Project packages: OK'); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m q1_supermimic.train --help
```

训练默认使用 `cuda:0`，需要可用的 NVIDIA GPU、兼容驱动及足够显存。`CUDA available: False` 时可以先检查包导入，并在取得本项目绕杆 checkpoint 后进行 CPU 评估和回放；默认 GPU 训练尚不具备运行条件。

## 准备权重与开始训练

按[资产与复现说明](pole-weaving.md)准备任务 7.7 warmup 文件、核对哈希，再运行：

```bash
# 只读预检查，不创建运行目录。
python -m q1_supermimic.train --check

# 一次更新的冒烟训练。
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json --max-updates 1
```

正式训练、恢复和评估命令见[绕杆任务说明](../courses/pole-weaving/README.zh-CN.md)。模型和训练产物不随源码分发。缺少 warmup 时预检查应明确失败，不得用随机初始化或其他任务权重替代。

## 测试与常见问题

在 README 指定的 `env_q1` 环境中补齐测试依赖并运行：

```bash
conda activate env_q1
python -m pip install -r requirements-test.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest
```

`requirements-test.txt` 复用固定的训练依赖，并补充 SciPy 和 Pinocchio（安装包名 `pin`），用于共享仿真的独立数值与 URDF 运动学对照。测试没有启动 Isaac Gym；G1 几何测试使用仓库已有的独立四元数工具，运行本测试集无需安装 Isaac Gym。

- 绕杆源权重测试需要登记的 warmup。所有测试从仓库根目录启动；共享仿真测试自行处理其源码目录和相对资源路径，不需要恢复旧的根目录软链接。
- 共享 Q1 测试包含真实 CUDA/Warp 检查。完整 GPU 验证应在 CUDA 可用的环境执行，不能把无 GPU 环境中的跳过当作 GPU 测试通过。
- `3rdparty/SuperMimic/README.md` 描述上游独立 Q1 V2 项目的 Python 3.13/uv 环境。本项目的 Q1 绕杆适配使用根 README 指定的 Python 3.11/Conda 环境，不应混装两套依赖。
- SuperMimic 默认将动作缓存写入用户的 `~/.cache/supermimic/motions`。测试和运行需要该目录可写；受限沙箱中的缓存权限错误不应通过修改训练算法规避。
- 依赖源码缺失时，从本仓库同一版本分支恢复；导入失败前先检查 `3rdparty/SuperMimic/src/supermimic/algorithm.py` 是否存在。
- `expert` 仅保留旧版绕杆所需的策略架构与配置；现行训练由 `pole_task` 的源任务流程执行。共享仿真包不是第二个受支持的训练任务入口。
