# V2_sicc 服务器运行

本分支包含当前本地已验证的实现、配置和测试。迁移无需沿用本机绝对路径；从服务器仓库根目录运行，使用已经配置好的 `env_q1`。本次仅发布源码，不启动服务器训练。

## 更新与导入路径

先检查服务器工作区；有未提交改动时先保存，不执行 reset/覆盖。首次检出远端新分支可以使用：

```bash
git status --short
git fetch origin
git switch --track origin/V2_sicc
git submodule update --init --recursive
conda activate env_q1
python -m pip install --no-deps -e .
python -c "import pole_task; print(pole_task.ROOT)"
```

如果本地已经有 `V2_sicc`，改用 `git switch V2_sicc` 和 `git pull --ff-only origin V2_sicc`。打印的模块路径必须是刚更新的服务器仓库，不能指向另一个旧 checkout。子模块固定为仓库登记的版本，不更新到上游最新提交。

## 源权重

warmup 权重由用户单独传输，放在仓库内：

```text
courses/pole-weaving/pole_task/assets/checkpoints/warmup-checkpoint_2000.pt
```

登记的 SHA-256 为：

```text
b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5
```

程序会严格校验。不要改 manifest 哈希来绕过检查，也不要换成其他任务/随机模型。机器人资产、参考数据及源配置已在 Git 中。

## 服务器 GPU 检查与首组训练

按服务器规定在获得 GPU 资源的计算节点/作业内运行以下命令，不在无 GPU 的登录节点启动训练。调度器和资源申请参数按服务器实际规则配置。本版本不自动提交调度作业。

```bash
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl python scripts/prepare_lr_replay.py
```

依次执行三组小规模 smoke 和三组完整 batch=512/horizon=32 更新检查。必须三组全部通过，才生成服务器自己的 `logs/q1-pole-weaving/lr-replay-v1/readiness.json`；不要复制本机凭据替代服务器检查。该过程不会开始正式训练。

检查通过后先运行 control：

```bash
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl python scripts/cam_local.py train \
  --config configs/lr-replay-v1/control.json \
  --output logs/q1-pole-weaving/lr-replay-v1/control/seed-0/attempt-1
```

完整说明和另外两组命令见 [LR_REPLAY_RUNBOOK.md](LR_REPLAY_RUNBOOK.md)；其中本机绝对 Python 路径在服务器替换为激活 `env_q1` 后的 `python`。每次只运行一个重计算任务。源码、依赖、配置在资源检查后保持不变；GPU或依赖变化后重新检查。

控制台不持续显示训练指标，主日志在该组 `seed-0/attempt-1.log`，状态在 `attempt-1.run.json` 和目录内 `run.json`。W&B 使用离线模式，不上传。完成后由用户告知，再分析结果，不自动开始下一组。

## 产物与版本边界

- Git 包含源码、测试、配置、运行与分析说明；不包含 `logs/`、`wandb/`、权重、视频和 `docs/cam/evidence/` 的本机证据文件。文档中的这些证据路径是本地留档位置，不是服务器必备输入。本机 `Agent.md` 也不发布。
- 旧 B0 实验与新版本分开。新三组各自从登记源权重初始化，不能用旧 B0 的 best/latest 冒充新实验续训。
- 本机与服务器共用总计七个正式槽位的预算；不能在两台机器重复同一槽位再选更好结果。诊断 GPU 检查不属于正式训练。
- 143项相关 CPU 回归、收尾后的38项针对性复测已通过，两者有重叠；发布不重新运行重计算测试。服务器运行环境与训练收益需以服务器实际检查/实验为准。
