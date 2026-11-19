# 当前运行入口

2026-11-20：后续使用 [LR_REPLAY_RUNBOOK.md](LR_REPLAY_RUNBOOK.md) 的新版本三组命令；下列 CAM 命令保留为历史说明，当前不启动候选或重跑旧 B0。

# 本地运行

Python 固定为 `/home/wangzihe/miniconda3/envs/env_q1/bin/python`，cwd 为仓库根。完整入口：

```bash
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl /home/wangzihe/miniconda3/envs/env_q1/bin/python scripts/cam_local.py smoke --config configs/cam/B0.json --output logs/q1-pole-weaving/cam-local/diagnostics/B0-smoke-new
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl /home/wangzihe/miniconda3/envs/env_q1/bin/python scripts/cam_local.py profile --config configs/cam/B0.json --output logs/q1-pole-weaving/cam-local/diagnostics/B0-profile-new
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl /home/wangzihe/miniconda3/envs/env_q1/bin/python scripts/cam_local.py train --config configs/cam/B0.json --output logs/q1-pole-weaving/cam-local/v2-initial/B0/seed-0/attempt-1
```

示例路径已经存在时必须选择新诊断目录，不能覆盖。正式 attempt 按 RUN_SLOTS.json 登记，不可自行重复命令扩张预算。候选正式训练要求配置绑定离线校准 JSON 及哈希，诊断参数不能直接用于正式训练。seeds1,2还需通过的 promotion_record。第二次正式尝试必须记录根因和语义等价证据到该槽位的 retry-justification.json；至多两次尝试。

每进程在目标目录旁写 `<run_id>.run.json` 和 `<run_id>.log`，正式目录内另有 run.json、identity.json、config.json、metrics.jsonl、latest.pt 和合格时才生成的 best.pt。进程持有本实验独占 compute lock。不会因进程退出0就判定任务验收通过。

停止只向已核对 command/run_id 的本任务 PID 发送 SIGTERM；训练在完整更新边界保存 latest。记录中断原因与重试资格后，使用同一配置及 `train --checkpoint <latest.pt>` 恢复到新的 attempt 目录；`--max-updates` 是绝对上限，正式入口只接受2500，诊断使用 smoke/profile。

新 checkpoint 保存 RNG/replay/优化器/阶段/环境预算。旧 checkpoint 可以评估；缺失完整训练状态时拒绝精确 resume，不伪造随机状态。首轮新实验初始化始终使用登记源 warmup，不能用其他组项目 checkpoint 代替。

评估入口：`scripts/cam_local.py evaluate --checkpoint <best.pt或latest.pt> --output <新目录> --device cuda:0`。使用 checkpoint 配置，输出指标、验收、轨迹和相同视角视频。原 evaluate/acceptance/play 也支持 `--config`，显式冲突会拒绝。

汇总：`scripts/cam_local.py summarize --output logs/q1-pole-weaving/cam-local/v2-initial`。诊断身份会被排除；同槽位多个评估结果需显式选择合格 attempt，禁止自动挑最好的一次。

运行期间保持源码和依赖冻结。日志、权重、W&B 离线数据不提交、不上传。CUDA/资产/资源缺失按实际阻塞报告；不改驱动、不停止其他应用、不用长 CPU 训练替代。

用户最新约定：后续训练由用户根据给出的 W&B 离线命令启动，用户告知完成后再分析；不持续轮询、不自动启动下一组。当前运行交接见 HANDOFF.md。
