# 当前状态：lr-replay-v1 实现，等待用户运行 GPU 检查

用户批准 V1 四段采样、最近两批 replay；三组 control/low_lr/actor_cosine 共用刷新机制与质量门槛。新版本及手动 W&B 命令见 [LR_REPLAY_RUNBOOK.md](LR_REPLAY_RUNBOOK.md)。没有自动启动新正式训练；GPU 检查成功才冻结源码/依赖/配置并生成 readiness。旧 B0 及冻结源码保留，CAM 候选暂缓。总七槽位预算中，旧 B0 占一槽，新三组占三槽，剩三槽未分配。

本次修复了旧交接中记录的 evaluate 显式配置转发、重复 summarize 日志、run_id 和 --device 冲突处理；旧训练身份仍绑定旧冻结源码。CPU 验证详情见 TASK_BOARD.md 和 evidence/lr-replay-v1/。收到用户完成通知后再分析，不盯跑、不自动串行训练三组。

---

以下为历史交接，状态以本页顶部为准。

# 当前状态：B0已完成并完成结果分析

已核验 B0 完成2500更新、退出码0。best.pt 为2075，最新独立CPU五用例及20次固定nominal均通过；latest.pt为2500，五用例均摔倒。正式CAM候选未训练，后续训练仍需先提供W&B命令由用户启动。

结果见 IMPLEMENTATION_REPORT.md 和 evidence/b0-results/training-analysis.json。下一步应先结合固定replay（Actor阶段779条transition）和模型多步误差分析后期退化；是否调整训练机制需显式新版本/同机制B0_control，不能混用旧对照。若继续既定CAM协议，仍须先进行原定离线校准，不直接用诊断参数训练。

---

# 当前运行交接

用户最新要求覆盖此前自主长训练安排：后续需要训练时给出带 `WANDB_MODE=offline` 的完整命令，用户启动并告知完成。停止持续轮询，不自动接着启动 B_ref/B_task。

## 原交接快照（已完成，以下为历史记录）

- PID：46497；交接检查确认进程仍在运行。只停止了 Codex 监控循环，没有停止训练。
- 交接快照：update=2131/2500，Actor阶段；update=2125任务验收通过。快照不是当前实时状态或完成结论。
- 目录：`logs/q1-pole-weaving/cam-local/v2-initial/B0/seed-0/attempt-1`。
- 主日志：该目录旁的 `attempt-1.log`；过程记录为 `attempt-1.run.json` 和目录内 `run.json`。
- 正式保存点只有 `best.pt` 与 `latest.pt`，各约203 MiB，总约406 MiB；每次覆盖更新，不累计每个评估时点的历史文件。因此未调整保存频率或删除保存点。另有67.3 MiB的独立诊断模型，不是额外正式训练。
- 执行源码保持冻结：`evidence/frozen-source.json` 和日志实验根的 `frozen-source.tar.gz`。训练过程中不改运行源文件。

## 收到完成通知后

1. 只读核验 PID、退出码、stop_reason、完整2500更新与checkpoint身份，更新 RUN_SLOTS.json；不把启动成功当完成。
2. 按原选模协议选 best（无 best 则 latest 并说明），完成真实评测与质量/视频分析，核对评估JSON的checkpoint哈希。
3. B0不合格先分析；合格则用训练参考和B0训练用例轨迹做固定离线CAM校准，不用eval-01调奖励，不做优化器更新。绑定校准凭据到两个候选配置。
4. 提供后续校准后完整资源复核和候选训练的带W&B命令，等待用户运行、告知结果。不得直接执行下一组长训练。候选参数未校准前不能用当前诊断值启动正式训练。
5. 继续遵守七个预定义槽位、每槽位至多一次合格重试和候选晋级门槛；P6不实施。

## 后续收尾检查

已完成124项CPU回归、2项GPU恢复对照、三组小GPU smoke与512/32完整资源测量，评测/视频端到端演练通过。正式B0尚未结束，尚不能报告CAM收益。

最终交付前还应补充分肢体动作方向导数与上肢/分部CAM诊断。审阅薄CLI时发现需收尾的接口细节：cam_local evaluate当前没有转发显式--config（底层和原evaluate/acceptance/play已校验）；summarize重复使用同一目录会碰到独占日志文件；run_id目前取叶目录名，应以完整实验/组/seed/attempt限定；train的--device应明确以配置为准或拒绝显式冲突。当前执行的B0命令不触发这些问题。只能在没有活动训练时处理，并保留冻结实验源码及非训练语义等价证据，不能悄悄混用训练版本。
