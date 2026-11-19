# Q1 CAM v2.0 本地任务

用户批准计划为执行合同，开发包位于 `/home/wangzihe/下载/Q1_CAM_Local_Codex_Development_Pack_v2.0/Q1_CAM_Local_Codex_Development_Pack`。

| 阶段 | 状态 | 证据/下一步 |
|---|---|---|
| P0 | COMPLETE | evidence/p0-preflight.json；资产与 CUDA 可用 |
| P1 | CPU_PASS | evidence/p1-physics.txt；最大归一化误差1.86e-7 |
| P2 | GPU_SMOKE_PASS | 三组真实 Actor 更新通过；共享奖励、B0 数值和梯度通过 |
| P3 | CPU_PASS / GPU_PASS | 124项回归 + 2项GPU恢复；评测/视频端到端通过 |
| P4 | B0_COMPLETE | B0完成2500更新；B_ref/B_task正式训练未运行 |
| P5 | B0_ANALYZED | best=2075路线通过但动作质量未获用户接受；LR调度几乎恒定；见学习率诊断，CAM比较未完成 |
| P6 | NOT_AUTHORIZED | 不实施 |

首轮仅 B0/B_ref/B_task seed=0。只有候选通过预登记标准，才运行 B0 与一个候选 seeds=1,2。所有组各自从登记源 warmup 初始化，项目 2000 warmup + 500 Actor。诊断与正式槽位分开记账。

## 旧运行交接与完成核验（历史）

用户要求后续训练仅提供带 W&B 离线记录的完整命令，由用户启动并告知结束；不自动启动下一组，不持续盯跑。当前已启动 B0 保持运行。本次交接快照为 update=2131/2500、Actor阶段，update=2125验收通过；该数字仅为交接时快照。详见 HANDOFF.md。

最新完成核验：B0退出0、recipe_exhausted、2500条连续更新；源码与checkpoint哈希通过。具体数据见 IMPLEMENTATION_REPORT.md 的 B0结果章节。后续训练仍由用户启动。


## 当前：lr-replay-v1（2026-11-20）

| 工作 | 状态 | 证据 / 下一步 |
|---|---|---|
| V1 采样移植 | IMPLEMENTED / CPU_PASS | 四段起点、25更新刷新、最近两批；V1 文件哈希及等价测试 |
| 三组 Actor LR | IMPLEMENTED / CPU_PASS | control=1e-5、low_lr=3e-6、actor_cosine=1e-5→1e-6；Model/Critic 不变 |
| 质量测量/选模 | IMPLEMENTED / CPU_PASS | 足滑/相对前倾/PD目标抖动；同种子测量开关物理奖励一致 |
| 精确恢复 | CPU_PASS | 跨阶段、刷新前后；模型/优化器/replay/RNG逐项相等 |
| GPU 新资源路径 | PENDING_USER | scripts/prepare_lr_replay.py；未自动运行 |
| 正式三组 / 收益 | NOT_RUN / UNKNOWN | 用户手动逐组运行；完成后分析，不盯跑 |

本版本不引用旧 GPU 测试作为新路径通过的证据。旧 B0 + 新三组共占四个预定义槽位，总预算七个不变；原 CAM 候选及扩种子暂缓。当前命令与门槛见 LR_REPLAY_RUNBOOK.md。

验证记录：143项相关 CPU 回归通过；保存点边界收尾后38项针对性复测通过（含重叠测试，不相加）。三组 CPU preflight 的资产/严格加载均通过；沙箱内 CUDA 不可见，新版本 GPU 关卡仍待用户执行。
