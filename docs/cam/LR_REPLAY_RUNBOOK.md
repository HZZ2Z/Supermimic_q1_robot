# lr-replay-v1：学习率与 V1 采样试验

2026-11-20。用户批准采用 V1 四段起点采样与最近两批 replay。实现与 CPU 验证完成状态见 TASK_BOARD.md；GPU 新路径及正式训练须由用户执行下面命令。旧 v2-initial/B0 保留，不混合作为严格对照。CAM 候选暂缓。

## 固定的三组

| variant | Actor 学习率 | Actor 调度 |
|---|---:|---|
| control | 1e-5 | 恒定 |
| low_lr | 3e-6 | 恒定 |
| actor_cosine | 1e-5 → 1e-6 | 500 次 Actor 更新内余弦下降 |

三组分别从登记的同一源 warmup 初始化，再执行项目 2000 warmup + 500 Actor。warmup Actor 冻结。Model/Critic 学习率、优化器、原奖励（含手臂 0.05）、单 Actor、网络、79/78/570/149/22、FP32、horizon=32 不变。仅独立版本化的 local_training_profile 改变 Actor 学习率/采样/选模；原 source manifest/TOML 及 source_training 硬校验保持原值。首选 batch=512，三组完整资源检查全部通过才生成 readiness；OOM 则停止并反馈日志，不自行改变 horizon、精度或单独降低某组 batch。

control 的恒定率与旧 B0 的几乎恒定 source cosine 仍有细小区别。因此旧 B0 与 control 的变化不能严格单独归因于 replay；新三组之间才隔离 Actor 学习率机制。此处未设置额外训练搜索。

## V1 采样语义

warmup 使用原来的均匀起点采集一次。Actor 在 update=2001、2026、2051 … 2476 采新数据；nominal 每批 4 条、straight/left_turn/right_turn 各 2 条。合法起点 1..last_frame-10 分成四段；nominal 一批覆盖四段，retention 两批轮换覆盖四段。保留 V1 的第一随机数及 seed=实验seed+1000+update+episode_index 规则。

Actor 首批丢弃 warmup replay；以后保留最近两批完整采集数据。episode、batch 边界不会被时间窗跨越。每批长度取决于剩余路线/终止，四段起点并不代表四段 replay 时间窗等权。collection.jsonl 记录起点、有效窗口帧；metrics.jsonl/W&B 记录各用例 replay transition 数、实际 model/policy 抽样数。只计新增真实交互，合并 replay 不重复扣预算。

checkpoint 保存 replay、批次边界、采集计数、RNG、优化器和已完成更新。同阶段与刷新边界恢复不额外重采样。不同 variant/资源档位/配置或源码拒绝 strict resume；不得把旧 B0 权重当作新实验续训。

V1 来源是本地导出目录 ../V1，无 .git；文件哈希见 evidence/lr-replay-v1/v1-provenance.json。V1 的 B2400 单种子通过记录支持采样可行性，但它还使用额外足滑/触地奖励，且后期曾退化，不能据此保证本轮收益。本轮不移植这些奖励。

## 统一质量门槛与 best

保留五用例原路线、碰撞、跌倒和 root XY 门槛，再要求：

- 足部承重 >10 N 时，接触点切向速度按力加权、按足部承重时间聚合。均速 ≤0.10 m/s、速度 >0.05 m/s 的承重时间比例 ≤0.20；左右脚必须均有承重记录。使用 V1 的 pre-integration qvel/接触 Jacobian 时序。
- 根部前倾相对下一帧参考的误差 RMS ≤5°，避免把参考本身的前倾当异常。
- PD 目标关节加速度 RMS / max(参考关节加速度 RMS,10 rad/s²) ≤2；目标为 q_default+0.5×action，用二阶时间差分计算，不跨 reset。
- 根姿态 SO(3) RMS 必须有限；所有门槛对五个用例统一适用。

后两项阈值是本次预登记的工程诊断门槛，不是 V1 已证明的标准或硬件安全限值。若无人通过，诚实报告无合格 best，不在看过候选后追调门槛。仅合格时按 nominal-01/eval-01 等权根姿态 SO(3) RMS 最小选 best；eval-01 始终是参与选模的验证用例。

足滑新指标 foot_slip_mean_m_s / foot_slip_fraction 不等同于旧事后几何接触 foot_slip_mps RMS（旧 B0 约0.34 m/s）；评测同时保留两者，禁止直接用不同定义的数值计算改善率。动作加速度是指令抖动代理，仍需真实视频检查身体运动。

最多保留 warmup.pt（项目第2000更新）、latest.pt、best.pt 三份权重，均为有完整状态的保存点；best 不保证一定存在。旧 B0 权重不删除。只在完整更新边界保存，保存到临时文件后原子替换。W&B 离线数据只落本地，不上传。

## 用户执行

工作目录与 Python 固定。下面准备命令依次执行三组小 smoke 和三组完整训练步骤资源检查（包括 warmup、Actor、两批 replay Actor），不会自动开始正式训练。检查日志有独立 PID/命令/状态；全部通过后归档源码/配置，记录依赖，生成 readiness.json。改动源码、依赖、配置会使凭据失效。

```bash
cd /home/wangzihe/公共/Supermimic_q1_robot/V2
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl /home/wangzihe/miniconda3/envs/env_q1/bin/python scripts/prepare_lr_replay.py
```

准备成功后，先运行 control；完成后告知 Codex 分析，不自动启动下一组：

```bash
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl /home/wangzihe/miniconda3/envs/env_q1/bin/python scripts/cam_local.py train --config configs/lr-replay-v1/control.json --output logs/q1-pole-weaving/lr-replay-v1/control/seed-0/attempt-1
```

low_lr、actor_cosine 命令已预登记，后续逐组手动执行，同时只运行一个重计算任务。control 若数值/物理失败，先排查；若仅质量门槛失败，不把它当合格基线，后续低 LR 诊断需单独分析。

```bash
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl /home/wangzihe/miniconda3/envs/env_q1/bin/python scripts/cam_local.py train --config configs/lr-replay-v1/low_lr.json --output logs/q1-pole-weaving/lr-replay-v1/low_lr/seed-0/attempt-1
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl /home/wangzihe/miniconda3/envs/env_q1/bin/python scripts/cam_local.py train --config configs/lr-replay-v1/actor_cosine.json --output logs/q1-pole-weaving/lr-replay-v1/actor_cosine/seed-0/attempt-1
```

每组目录旁 attempt-1.log 是主日志，attempt-1.run.json 记录命令/PID/退出状态；目录内 metrics.jsonl、collection.jsonl、config.json、identity.json、run.json、stop_reason.json、wandb/ 和权重为完整产物。日志由进程直接写入，终端不会持续输出训练指标。不要重复启动同一命令。

回放（有合格 best 时）：

```bash
/home/wangzihe/miniconda3/envs/env_q1/bin/python -m pole_task.play --checkpoint logs/q1-pole-weaving/lr-replay-v1/control/seed-0/attempt-1/best.pt --case nominal-01 --device cuda:0 --gui
```

完整五用例/20次 nominal/指标/视频评估，默认恢复保存配置：

```bash
WANDB_MODE=offline OMP_NUM_THREADS=4 MUJOCO_GL=egl /home/wangzihe/miniconda3/envs/env_q1/bin/python scripts/cam_local.py evaluate --checkpoint logs/q1-pole-weaving/lr-replay-v1/control/seed-0/attempt-1/best.pt --output logs/q1-pole-weaving/lr-replay-v1/control/seed-0/evaluation-best --device cuda:0
```

无 best 时可显式评估 latest.pt，但不能称其为质量验收通过。停止/恢复沿用 LOCAL_RUNBOOK.md 的更新边界机制；重试记录须包含根因与资格，最多 attempt-2，不使用重试名义改语义。

## 预算与结论边界

已消耗旧 B0 seed0 一个槽位；本版预登记 control/low_lr/actor_cosine seed0 三个槽位，总计四个，最多七个的总预算不增加。剩余三个暂不分配；原 CAM 候选和扩种子计划推迟，不能另开七个槽位。诊断资源检查另计，不计为合格正式 run。

收到完成通知后分析：实际 Actor 学习率、早期/后期任务与质量曲线、replay 覆盖、梯度、世界模型误差、best/latest 回放。低 LR 改善只支持这一设置下的证据；没有收益也如实报告。单种子结果不声称统计显著性，不声称独立测试泛化。
