# 足底奖励与验收

2026-11-15：修正足地摩擦随机化，增加 foot slip 和近地落脚速度辅助惩罚，接入实际接触滑移统计与选模。没有启动训练或更新策略权重。

## 修改位置

| 文件 | 内容 |
|---|---|
| `courses/pole-weaving/pole_task/source_environment.py` | 脚和地面同时采用采样摩擦；在每个物理子步记录足地接触速度。 |
| `3rdparty/SuperMimic/src/supermimic/robot.py` | `foot_kinematics()` 读取当前模型每脚四个碰撞球，计算球底材料点的位置与速度。 |
| `3rdparty/SuperMimic/src/supermimic/reward_tuning.py` | 两项辅助惩罚及参数校验；与基础奖励复用运动学结果。 |
| `courses/pole-weaving/pole_task/source_runtime.py` | `SOURCE_REWARD_TUNING` 启用两项奖励，仿真采样和想象训练共用；汇总回合指标。 |
| `courses/pole-weaving/pole_task/foot_contact.py` | 实际滑移的测量阈值与验收函数。 |
| `courses/pole-weaving/pole_task/config.json` | 滑移验收上限。 |
| `courses/pole-weaving/pole_task/source_training.py` | 路线和滑移共同验收、归一化选模评分、恢复训练一致性检查。 |
| `play.py`、`evaluate.py`、`acceptance.py` | 输出并使用新的滑移结果。 |

上述任务文件均位于 `courses/pole-weaving/pole_task/`，共享机器人与奖励代码位于 `3rdparty/SuperMimic/src/supermimic/`。

## 与原有奖励模块的关系

本次修改前已经存在 `RewardTuningCfg`、`reward_tuning_terms()` 和 `tuned_reward()`，原任务用它们启用权重为 0.05 的手臂关节惩罚。足底两项沿用同一配置和汇总函数，没有新增训练奖励脚本或第二套奖励入口。

当前训练入口 `pole_task.training.main()` 转交 `source_training.main()`。其中，仿真采样 `run_source_episode()` 与世界模型想象 `SourcePoleSuperMimic._imagined_step()` 都调用同一个 `tuned_reward()`，并由训练端传入同一个 `SOURCE_REWARD_TUNING`：

```text
仿真采样 / 世界模型想象
  → tuned_reward()
      → Q1Robot.reward()：原有基础奖励，已乘 dt
      → reward_tuning_terms()：手臂项 + 足底滑移项 + 近地落脚项
  → 基础奖励 + dt × 辅助项之和
```

`Q1Robot.reward()` 原有的全身位置、姿态、线速度、角速度跟踪，根部位置／姿态跟踪，动作平滑与关节越限公式均保留。其接口新增了可选的关键字参数 `next_heads`，仅复用本步运动学计算，与已有的 `target_heads` 缓存对应；旧调用方式仍可使用。这是本次明确的接口扩展，不应表述为“接口完全未改”。`foot_kinematics()` 负责从同一运动学结果提取足底点位置和速度，惩罚公式仍在 `reward_tuning_terms()` 内。

新增的 `foot_contact.py` 只保存实际接触测量阈值、验收门槛检查与版本标识，不计算奖励。实际接触数据由 `SourcePoleEnvironment` 汇总后用于评估和选模；可微奖励继续基于预测状态计算。旧的 `models.PoleRobot.reward()` 属于保留的旧任务实现，当前 source-profile 训练使用继承 `Q1Robot.reward()` 的 `SourcePoleRobot`。

`test_pole_source_runtime.py` 在仿真采样和想象两条路径中分别关闭／开启足底项，确认状态相同且奖励差恰好等于 `dt × (foot_slip_penalty + foot_landing_penalty)`。两条测试都不执行优化器更新。

## 奖励初值

| 参数 | 初值 | 单位／意义 |
|---|---:|---|
| `foot_slip_weight` | 0.05 | 滑移惩罚权重，在 `SOURCE_REWARD_TUNING` 设置。 |
| `foot_landing_weight` | 0.02 | 近地下降速度惩罚权重，同上。 |
| `foot_slip_speed_std` | 0.25 | m/s，滑移速度归一化尺度。 |
| `foot_landing_speed_std` | 0.5 | m/s，下降速度归一化尺度。 |
| `foot_contact_height` | 0.01 | m，足底近地支撑权重的衰减高度。 |
| `foot_landing_height` | 0.04 | m，近地落脚速度约束的衰减高度。 |

尺度和高度的默认值位于 `RewardTuningCfg`，也可在 `SOURCE_REWARD_TUNING` 中显式覆盖。权重为零可关闭对应奖励。

足底点速度为 `v_origin + omega × offset_world`。滑移惩罚取每脚四个球底点的水平速度平方，在八个点上平均。支撑权重取实际与参考近地权重的最大值，并停止该权重本身的梯度，避免出现直接通过抬高脚来减小惩罚的梯度；参考支撑阶段也能约束预测中悬空的脚。实际和参考都处于离地摆动阶段时不施加滑移惩罚。

落脚项只惩罚近地的向下速度，向上抬脚不受该项惩罚。两项在 `tuned_reward()` 中统一乘控制周期 `dt`，不会重复缩放。

这是一种适用于当前平地模型的接触近似，并不等于接触力预测。实际碰撞还包括脚部 mesh，因此评估采用真正的物理接触，不能用代理奖励值代替滑移指标。模型、观测和动作维度没有增加。

## 实际滑移指标

每个 2 ms 物理子步只统计有效 foot–floor 约束。以单脚法向力 >10 N 作为承重条件，在同一只脚内部按接触法向力加权接触点切向速度，再按承重时间汇总两只脚：

- `foot_contact_time_s`：左右脚各自的承重秒数；双支撑会同时计时。
- `foot_slip_mean_m_s`：承重接触点滑移速度均值。
- `foot_slip_rms_m_s`：承重接触点滑移速度 RMS。
- `foot_slip_fraction`：承重时间中，单脚力加权滑移速度 >0.05 m/s 的比例。

零接触时速度／比例返回 `null`，不会被当成零滑移通过验收。每次 reset 清空累计值。单步积分时使用与接触 Jacobian 一致的积分前 qvel；不支持 RK4 的该采样方式。

## 验收与 best.pt

`config.json` 当前实验门槛：

```json
"foot_slip_mean_max_m_s": 0.1,
"foot_slip_fraction_max": 0.2
```

除原有路线、摔倒和碰杆要求外，所有要求的评估场景还必须满足滑移均速 ≤0.1 m/s、超速比例 ≤20%，且双脚都存在承重记录。缺失、非有限或无接触的滑移数据不能通过。

`route_success` 单独表示原有路线验收结果；回放和逐场景评估的 `success` 加入滑移要求。正式完整验收继续使用配置中的路线／保持测试门槛。

通过验收的候选按下式选择 best，分数越低越好，各场景等权：

```text
mean_over_cases(
    root_xy_rmse / corresponding_route_rmse_limit
    + foot_slip_mean / foot_slip_mean_max_m_s
    + foot_slip_fraction / foot_slip_fraction_max
)
```

若没有模型通过新门槛，仍会保存 `latest.pt`，但不会产生通过验收的 `best.pt`。这些是实验初值，不是实机安全标准。

新 checkpoint identity 记录 `foot_contact_version` 和完整 `reward_tuning`。旧训练 checkpoint 仍可加载做回放／评估；恢复训练时，缺少这些字段或奖励参数不同会明确拒绝，以免沿用旧分数和旧物理语义。后续新训练仍从登记的 warmup 初始化，无需重建网络结构。

## 验证范围

验证覆盖实际摩擦系数、足底点 Jacobian 速度、奖励值与梯度、摆腿豁免、近地下降约束、dt 缩放、缓存一致性、测量累计与 reset、验收／选模和恢复检查。另以登记的 warmup 世界模型检查了辅助奖励到动作的有限非零梯度，没有执行 optimizer step。

旧策略的五场景无界面评估结果保存在 `outputs/foot-contact-change-validation/evaluation.json`：全部保持路线成功，但未通过新滑移门槛。`nominal-01` 的原路线数值完全一致，滑移均速为 0.17266 m/s，超速比例为 30.48%。这是评估接线验证，不是新奖励已经改善策略的证据。

未执行训练、实机验证或 GUI 崩溃排查。训练收益、最优权重和可迁移性仍需后续实验确认。

## Actor 数据刷新与诊断记录（2026-11-15）

针对 `run-20261115-061744` 的后期退化，训练循环增加以下记录和采样行为，奖励公式、权重与 2,000 + 500 次更新预算保持不变：

- `actor_collection_interval: 25`：在第 2,001 次进入 Actor 阶段时采样，此后在 2,026、2,051……2,476 次更新前，用当前 Actor 重新采集训练场景并替换 replay。沿用既有场景数量和随机化，不加入留出场景或评估轨迹。恢复训练时先重新采样一次，再按原阶段的更新编号安排刷新。
- `actor_checkpoint_interval: 50`：保存 `checkpoint-002050.pt`、`checkpoint-002100.pt`……`checkpoint-002500.pt`，包含模型、优化器和原有恢复信息；提前达到调用方更新上限时也保存 Actor 快照。默认全程 10 份，按目前文件大小额外约占 2 GB。`best.pt` 继续要求完整验收通过，诊断快照不代表合格。
- 分项日志写入原有 `metrics.jsonl`，记录于评估或快照更新。`imagined_base_reward`、`imagined_arm_joint_penalty`、`imagined_foot_slip_penalty`、`imagined_foot_landing_penalty` 已包含权重和 `dt`，并使用与 `imagined_reward` 相同的存活权重平均；四者之和等于总想象奖励（允许浮点误差）。基础项为总奖励减去所有辅助项，不拆改基础奖励公式。
- 分项值通过已有 `reward_tuning_terms()` 在 `no_grad` 下额外计算，仅用于日志，不参与目标函数或反向传播；非记录更新不增加该计算。
- 同时记录本次运行的累计 `collection_environment_steps`、当前 `replay_transitions` 和 `last_collection_update`，以区别训练数据量与评估环境步数。`environment_steps` 在日志写入时包含当次评估。

两个间隔必须为正整数。快照间隔与评估间隔不一致时，快照可没有 `evaluation`；需另外评估该快照。恢复检查仍要求配置一致，旧 1744 运行也已用完 2,500 次预算，不能直接用旧 `latest.pt` 继续此配置。
