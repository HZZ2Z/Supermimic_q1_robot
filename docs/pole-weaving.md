# Q1 杆间绕行

当前主线为 **V1（B-2400）**：Actor默认四段采样，warmup仍均匀采样。[版本与模型说明](versions/V1.md)。

杆间绕行任务是独立的 SuperMimic 配置，用于在本仓库的足球场景中复现源任务 7.7 的三杆
S 形路线。它有独立且与源任务兼容的观测、重置、终止条件、Actor、世界模型和 Critic
接口；训练运行目录位于 `logs/q1-pole-weaving/`，checkpoint 格式也由本项目单独管理。
球和球门仅作为可见场景元素，不会进入杆间绕行任务的观测、奖励、重置或终止条件。

当前奖励在源任务基础上增加足底滑移和近地落脚速度惩罚，并修正足地摩擦随机化。
评估与选模同时检查路线和足底滑移；参数、指标定义和旧 checkpoint 的恢复限制见
[足底奖励与验收说明](foot-contact-rewards.md)。

## 所需资产

杆间绕行任务不再需要旁边的 `../../SuperMimic` 仓库副本。源 Q1 XML 和训练配置快照已保存在
本仓库中；XML 使用 `shared/q1-mujoco/resources/` 下已有的机器人网格。路线参考数据保存在
`courses/pole-weaving/pole_task/data/references/`。`courses/pole-weaving/pole_task/data/manifest.json` 中的绝对路径字段 `source`、
`source_catalog` 和 `source_geometry` 用于记录原始位置；程序运行时改用相对于仓库的 `file` 字段。

只有从源 warmup 开始训练时才需要外部提供模型文件。请联系项目维护者
获取 `checkpoint_2000.pt`，然后将其复制到登记路径：

```bash
mkdir -p courses/pole-weaving/pole_task/assets/checkpoints
cp /path/to/checkpoint_2000.pt courses/pole-weaving/pole_task/assets/checkpoints/warmup-checkpoint_2000.pt
sha256sum courses/pole-weaving/pole_task/assets/checkpoints/warmup-checkpoint_2000.pt
```

文件的 SHA-256 必须为
`b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5`.
训练前的预检查会核对该哈希。模型文件已加入 Git 忽略列表。源任务的最终 checkpoint 仅以哈希
记录来源，不需要复制到新主机。

`courses/pole-weaving/pole_task/source_manifest.json` 记录源代码版本和迁移文件的哈希。标记为
`required_for: ["provenance"]` 的文件仅作为历史来源记录，不需要原始仓库副本：

| 资产 | 登记路径 | 用途 |
| --- | --- | --- |
| `warmup_checkpoint` | `courses/pole-weaving/pole_task/assets/checkpoints/warmup-checkpoint_2000.pt` | 严格用于训练初始化；通过邮件索取 |
| `passed_final_checkpoint_reference` | 源任务 7.7 最终 checkpoint 的哈希 | 仅记录来源；不需要该文件 |
| `warmup_config`, `actor_config` | `courses/pole-weaving/pole_task/data/source-configs/` | 本地训练配置快照 |
| `robot_xml` | `shared/q1-mujoco/resources/q1_description/source_task_7_7/q1_v3_mc.xml` | 本地源机器人模型 |
| 路线目录、几何数据和参考轨迹 | `courses/pole-weaving/pole_task/data/` | 本地任务输入 |
| 源模型、路线和训练文件 | manifest 中记录的上游版本和哈希 | 仅记录来源；不需要原仓库副本 |

程序先核对登记的 warmup 文件哈希，再通过 `strict=True` 加载 Actor、世界模型、Critic 和机器人的
全部张量，并检查源任务测试样例的一致性。如果 warmup 权重缺失、损坏或无法识别，预检查会指出
不匹配的字段并停止；不会创建运行目录，也不会改用其他 checkpoint。源任务最终 Actor 和足球专家
checkpoint 都不会作为替代权重。本项目训练得到的 checkpoint 可在没有源 warmup 权重或上游仓库的
情况下进行评估和回放。

## 数据划分

`courses/pole-weaving/pole_task/data/manifest.json` 将 `nominal-01`（675 个控制步）和留出的 `eval-01`
（671 个控制步）与其有序杆位布局对应起来。留出的参考轨迹不会进入训练批次，只用于评估。
`straight`、`left_turn` 和 `right_turn` 是训练及固定步态保持测试使用的参考轨迹。每条参考轨迹和
路线布局都记录了对应哈希。

## 预检查与训练

预检查为只读操作：它会验证数据对应关系及哈希、源文件清单、warmup 严格加载结果、解析出的
2,000 + 500 次更新配置、输出目录隔离情况和硬件状态。预检查不会打开查看器或创建运行目录。

```bash
python -m pole_task.training --preflight-only --config courses/pole-weaving/pole_task/config.json
```

训练只会向杆间绕行配置的输出目录写入文件。每次运行都会记录源代码版本、warmup 路径和哈希、
解析后的训练阶段、优化器和学习率配置、数据划分、随机种子、环境步数预算、指标、停止原因及选用的
checkpoint：

```bash
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json
```

训练配置固定为：先冻结 Actor，在确定性物理环境中进行 2,000 次 warmup 更新；再开启随机化并冻结
每帧历史中的 6 个杆相关输入，进行 500 次 Actor 更新。每次更新都会按登记的批次大小和预测时长拟合
世界模型，然后拟合 Critic；在 Actor 阶段，还会通过与源任务对齐的想象过程拟合 Actor。机器人实际
碰杆时，该回合会终止，并作为 replay 边界和失败轨迹处理。

旧模型仍须指定其原配置：V1归档模型使用原B配置；旧主线配置已保存在 `experiments/reward-sampling/pre-V1-main.json`。新版默认配置不会自动兼容旧哈希。

恢复训练前会验证 checkpoint 格式、源 warmup 哈希、机器人和数据身份、配置及训练方案；验证通过后，
恢复本项目的 Actor、世界模型、Critic、优化器、更新序号、训练阶段和最佳分数：

```bash
python -m pole_task.training --config courses/pole-weaving/pole_task/config.json \
  --resume-checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt
```

checkpoint 的 `stage` 表示已完成更新所属的阶段：第2000次仍是warmup，恢复后从2001进入Actor。
第0次按初始阶段校验。warmup边界恢复使用默认学习率，不传 `--resume-actor-lr-scale`；该覆盖仍只接受
保存时已处于Actor阶段的checkpoint。

`success_patience` 只统计Actor阶段未取得更好验收分数的评估次数，warmup不累计或触发此停止条件；
进入Actor时从零开始，Actor取得更好分数时清零。环境步数预算和总更新上限仍独立生效。
新checkpoint在identity中记录 `patience_scope: "actor"`，恢复时保留此计数。
旧checkpoint没有该字段，无法可靠分离warmup与Actor累计值，恢复时仅将patience清零，并记录
`legacy_patience_reset: true`、`initial_actor_patience: 0`；最佳分数、更新编号和优化器状态不重置。
因此旧2100的历史patience=84不会继续扣减新运行的Actor额度。新格式后续恢复不会再次自动清零。

如需限制试运行时长，可添加 `--max-updates N`。默认预算2500；仅Actor检查点恢复时可用 `--actor-end-update N` 显式扩展预算，并记录到identity。`--run-name`可指定独立目录名，同名拒绝覆盖。如果
`best.pt` 存在，就使用它；否则使用 `latest.pt`。程序不会搜索足球或源任务最终 checkpoint。

### 从 2100 做学习率对照

以下历史模型使用原主线配置快照 `experiments/reward-sampling/pre-V1-main.json`，不能用V1新版默认配置恢复。

以下两组学习率对照已完成，对应 `run-20261115-221026` 和 `run-20261115-222238`。
它们当时使用单批replay。当前代码已改为Actor保留最近两批，重新执行这些命令不再复现旧的数据保留策略；
下一轮使用下方“两批replay续训”的原学习率命令。

`--resume-actor-lr-scale` 仅允许从 Actor 阶段的项目 checkpoint 续训，取值为 `(0, 1]`。
它乘在原有余弦调度后的 Actor 学习率上；世界模型、Critic、优化器状态、奖励和选模门槛不变。
来源配方与配置仍严格校验，不需要修改 `config.json`。省略该参数时继承 checkpoint 的缩放值，
旧 checkpoint 默认 1；显式传 1 可恢复登记的 Actor 学习率。

以下两组都从原 2100 开始，到全局更新 2200 停止，即各执行 100 次更新。
使用相同配置种子与采样规则，但策略分化后实际轨迹可能不同。两组顺序运行，避免争抢 GPU。

```bash
conda activate env_q1
cd /path/to/Supermimic_q1_robot
q1_config="experiments/reward-sampling/pre-V1-main.json"
q1_parent="版本记录/历史训练归档/run-20261115-160417/checkpoint-002100.pt"

# 先确认 hardware.ready 为 true；此命令不训练
python -m pole_task.training --config "$q1_config" --preflight-only

# 对照组：Actor 基础学习率 1e-5
python -u -m pole_task.training --config "$q1_config" \
  --resume-checkpoint "$q1_parent" --max-updates 2200 --resume-actor-lr-scale 1.0

# 实验组：Actor 基础学习率 5e-6，仍从原 2100 开始
python -u -m pole_task.training --config "$q1_config" \
  --resume-checkpoint "$q1_parent" --max-updates 2200 --resume-actor-lr-scale 0.5
```

每组自动创建独立运行目录；`identity.json` 和 checkpoint 记录父模型路径／哈希、
`resumed_from_update`、`actor_learning_rate_scale`、`effective_actor_base_learning_rate` 和
`requested_max_updates`。`recipe` 保留登记的基础配方，`metrics.jsonl` 中的
`actor_learning_rate` 是实际调度后的值。预检查仍检查原有配置与硬件；父 checkpoint 和缩放参数
会在正式入口创建运行目录前验证。正常完成时评估 2125、2150、2175、2200，编号快照保存 2150、2200。
原有环境步数和 patience 停止条件仍有效；不保证产生通过验收的 `best.pt`。

原指标日志增加三项分场景统计，不改变采样：

- `replay_case_transitions`：当前 replay 中各场景的转移数；现行Actor路径包含最近两批。
- `sampled_model_windows_by_case`：本次运行累计实际抽到的世界模型训练窗口数。
- `sampled_policy_windows_by_case`：本次运行累计实际抽到的策略／Critic 批次窗口数。

窗口数统计有效采样的次数，包含重复抽样，不是独立数据量，也不重复计算 Critic 对同一批次的多轮拟合。
两个累计值在新运行目录从零开始，可对相邻日志做差观察覆盖变化；`eval-01` 应始终为零。
保持每 25 次数据刷新和评估、每 50 次快照，优先比较 2150／2200 的逐场景分阶段滑移与视频，
不因左转改善就忽略其他场景退化。

### 从 2100 做两批 replay 续训

本轮只改变Actor数据保留方式，并增加连续承重／再承重诊断。原学习率、奖励和正式选模门槛不变。
先执行只读预检查，在训练终端确认 `hardware.ready: true`：

```bash
conda activate env_q1
cd /path/to/Supermimic_q1_robot
python -m pole_task.training --preflight-only \
  --config "courses/pole-weaving/pole_task/config.json"
```

确认后再运行：

```bash
python -u -m pole_task.training \
  --config "courses/pole-weaving/pole_task/config.json" \
  --resume-checkpoint "版本记录/历史训练归档/run-20261115-160417/checkpoint-002100.pt" \
  --max-updates 2200 --resume-actor-lr-scale 1.0
```

这会从原2100执行100次更新，Actor基础学习率仍为1e-5，保留余弦调度。
初次采样批次为2101，后续replay批次应依次为 `[2101, 2126]`、`[2126, 2151]`、`[2151, 2176]`。
独立运行目录记录 `actor_replay_batches: 2`；正常完成时保存2150、2200快照。
可与单批原学习率运行 `run-20261115-221026` 对照，重点检查困难时段覆盖、连续承重速度及再承重事件，
并保留nominal、straight等场景的逐项比较。原停止条件仍有效，不保证生成 `best.pt`。

## 无界面评估与回放

以下命令默认以无界面模式运行，只接受本项目训练的杆间绕行 checkpoint。checkpoint 缺失、格式属于
足球任务、配置不匹配或属于源任务最终模型时，程序会拒绝加载，不会改用其他权重。

使用固定随机种子评估登记的常规、留出和步态保持用例：

```bash
python -m pole_task.evaluate \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
```

运行完整的无界面验收：测试全部登记用例，以及 20 次使用固定随机种子的常规路线试跑。常规和留出
用例必须到达终点 0.5 m 范围内，根部 XY 均方根误差（RMSE）不超过 0.5 m，且不能碰杆或摔倒；
步态保持用例不能摔倒，RMSE 不超过 0.3 m；20 次常规路线试跑必须全部通过。

```bash
python -m pole_task.acceptance \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
```

不打开显示窗口，回放一个登记用例：

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --case nominal-01
```

仅在有可用显示器时添加 `--gui`。添加 `--loops N` 可连续播放 N 次，每次的随机种子依次为
`--seed`、`--seed + 1`，以此类推。例如：

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt \
  --case nominal-01 --gui --seed 20261104 --loops 5
```

每次播放的路线完成情况、终点距离、运行步数、碰杆次数、摔倒情况、根部 XY RMSE 和成功状态，
都会写入 checkpoint 所在运行目录旁的结果文件。GUI 循环播放时会在同一个 MuJoCo 窗口内重置
仿真并开始下一轮，窗口两侧的工具栏默认关闭。
