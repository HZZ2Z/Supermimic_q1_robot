# 奖励与采样对照

A/B对照已完成；V1选用B-2400，B采样方式已成为主线默认。这里的A/B配置保持原样供历史恢复；旧主线字节快照为`pre-V1-main.json`。详见[V1](../../docs/versions/V1.md)。

A：两批replay＋原均匀随机起点，滑移权重0.05。
B：两批replay＋Actor四段轮换起点，滑移权重0.05。
C：仅在离线筛选通过时由筛选脚本生成；优先0.075，其次0.10。未通过不训练C。

实验建立时原默认config.json和奖励公式未改；V1发布后默认采样改为B，奖励公式仍不变。两组均从登记的源warmup初始化，课程warmup 2000次，Actor 200次，到2200停止。不要添加resume参数，也不要使用2100作为初始化。

新配置字段：`actor_start_sampling`（默认uniform，可选stratified_4）、`foot_slip_weight`（默认0.05，有限非负数）。有效权重沿采集、想象、评估与checkpoint身份传递，恢复仍严格验证配置和奖励。

四段仅在Actor启用。合法起点1…末帧−10分成四段，段号为(batch_index × episode_count + episode_index) % 4；batch_index由Actor阶段全局更新和25次刷新间隔计算。段内沿用回合种子的首个均匀随机数。warmup采样、物理种子和两批replay规则不变。

`collection.jsonl`记录每回合的批次更新、场景、起止帧、转移数，以及不跨边界的模型／策略窗口对应参考帧。它是候选窗口覆盖；实际抽样总数仍见metrics.jsonl，不能把候选窗口计为实际逐帧抽样。

## 离线筛选

```bash
conda activate env_q1
cd /path/to/Supermimic_q1_robot
python -B scripts/experiments/screen_reward_sampling.py --output outputs/reward-sampling-screen-new
```

脚本冻结三个checkpoint的网络，取四个训练场景各12个锚点，共144个，不使用eval-01。对32步目标的首动作求导，以0.01／0.02 RMS做正负扰动，复制MuJoCo完整状态、动作延迟与观测历史，闭环模拟0.20秒，再补40ms尾部用于接触分类。每个锚点额外重复零扰动回放，要求结果完全一致。记录1872条零／正／负扰动结果，禁止optimizer.step。

同一冻结网络中目标对滑移权重线性，因此复用各分项梯度计算三个候选方向。`predicted_first_order_objective_delta`是模型目标的一阶预测，不冒充完整非线性预测，也不是实际接触收益。梯度冲突由当前滑移代理项计算；旧Critic不代表新权重训练后的价值函数。

准入比较采用相同扰动幅度的正方向试验，对照权重0.05：左转连续均速下降≥5%或再承重积分下降≥10%，另一项不增；其他三场景两项不增；各场景事件数不增、承重时间变化≤5%、梯度冲突不增且无失败。两个幅度均须通过。负方向与零扰动只用于诊断，不事后改变准入标准。

## 获得明确训练授权后再运行

先在训练终端执行各组预检查，确认hardware.ready为true；不要并行抢GPU。

```bash
python -m pole_task.training --config experiments/reward-sampling/A.json --preflight-only
python -m pole_task.training --config experiments/reward-sampling/B.json --preflight-only
```

A、B顺序启动：

```bash
python -u -m pole_task.training --config experiments/reward-sampling/A.json --max-updates 2200
python -u -m pole_task.training --config experiments/reward-sampling/B.json --max-updates 2200
```

正常完成保存2050、2100、2150、2200，独立目录分别在logs/q1-pole-weaving/experiment-A、experiment-B。不要自动延长至2500。C.json不存在即未获准入，不创建占位配置。

每组四份快照均用 `python -m pole_task.evaluate --checkpoint <文件> --device cpu --output <独立JSON>` 评估五场景；CPU结果单独与原2100比较，不混用训练GPU结果。回放检查左转1.0–1.5、2.0–2.5、4.0–4.5、5.0–5.5秒及nominal／straight换脚。没有满足左转改善且其他场景无退化的候选就报告未胜出；正式best.pt规则不变。

当前实施不自动训练或推送GitHub。单种子与短扰动筛选不能代替多种子、长时闭环或实机验证。

## Actor预算扩展与命名

`--run-name`直接指定output_root下的目录名，并记录到identity.json和checkpoint。同名已存在时报错，不覆盖、不自动加后缀；省略仍使用时间戳。支持1–96个字母、数字、下划线、点和连字符，首字符不能是点或连字符。

`--actor-end-update`仅用于Actor检查点续训，显式扩展全局预算；不能缩短已登记的预算。默认仍为2500。扩展记录为identity.actor_end_update，原配置和recipe保留登记值并严格校验；奖励、学习率余弦调度、采样、正式验收条件不变。后续从扩展检查点恢复会继承预算；`--max-updates`仍可限制本次运行的停止位置。

A-2200继续到2600（新增400次Actor更新），在独立终端执行：

```bash
conda activate env_q1
cd /path/to/Supermimic_q1_robot
systemd-run --user --scope --unit=q1-a-to2600 \
  python -u -m pole_task.training \
  --config "experiments/reward-sampling/A.json" \
  --resume-checkpoint "logs/q1-pole-weaving/A-from1000-to2200/checkpoint-002200.pt" \
  --actor-end-update 2600 --max-updates 2600 \
  --run-name A-uniform-from2200-to2600
```

输出目录为`logs/q1-pole-weaving/experiment-A/A-uniform-from2200-to2600`。仍每25次评估、每50次保存编号检查点；原有patience和环境步数停止规则继续生效。不要与B并行占用内存/GPU。独立scope不能避免系统内存耗尽；恢复仍不保存RNG和replay，不能视为未中断运行的逐位复现。
