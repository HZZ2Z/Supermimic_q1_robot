# Q1 Robot Soccer · V1

基于 SuperMimic 的 Q1 三杆绕行。当前版本 **V1 = B实验第2400次更新的 best.pt**，CPU五场景验收通过5/5。

## 一、效果展示

### Baseline

[![Baseline策略演示](docs/assets/demos/baseline-nominal-01.gif)](docs/assets/demos/baseline-nominal-01.mp4)

### Foot slip · 原2100

[![Foot slip策略演示](docs/assets/demos/reward-footflip-20261115-2100.gif)](docs/assets/demos/reward-footflip-20261115-2100.mp4)

### V1 · B-2400

[![V1策略演示](docs/assets/demos/V1-B2400-nominal-01.gif)](docs/assets/demos/V1-B2400-nominal-01.mp4)

点击预览查看MP4。V1为一次完整CPU仿真推理，13.5秒，按仿真时间播放；旧视频来自录屏，画面速度不用于定量比较。

## 二、优化记录

**Foot slip**（历史分支名 `reward_footflip`）：修复摩擦随机化；沿用原奖励入口，加入足底滑移和近地落脚速度惩罚；将实际滑移纳入评估与选模。

**V1**：主线默认采用B的Actor四段轮换、段内随机起点，warmup仍均匀随机；Actor replay保留最近两批。每25次刷新／评估，每50次保存检查点，增加分场景采样和接触分阶段日志，并修复恢复阶段边界与提前停止计数。奖励权重和正式验收门槛不变。

同为 `nominal-01`、CPU、种子20261103：

| 指标 | Baseline | Foot slip 2100 | V1 |
|---|---:|---:|---:|
| 全接触滑移均速（m/s） | 0.1727 | 0.0910 | **0.0833** |
| 超速承重时间比例 | 30.48% | 16.83% | **13.58%** |
| 根部XY轨迹RMSE（cm） | 2.73 | 2.69 | **1.65** |
| 五场景正式通过数 | 0/5 | 4/5 | **5/5** |

V1相对原2100，左转连续承重均速下降36.9%、再承重积分下降54.0%。但左转和直行承重时间减少约7%，相对B-2200直行再承重积分仍增加13.6%。结果来自单种子固定仿真，尚未做实机验证。

[完整V1说明](docs/versions/V1.md) · [五场景指标](docs/versions/V1-evaluation.json) · [版本身份与哈希](docs/versions/V1.json)

## 三、使用

先按[环境与资产说明](docs/setup.md)准备 `env_q1` 和登记的源warmup权重。项目权重、原始录像和日志不随Git分发；本机V1已归档到 `版本记录/V1/`。

```bash
conda activate env_q1

# 回放V1；--loops 1只播放一次。
python -m pole_task.play --checkpoint "版本记录/V1/best.pt" \
  --case nominal-01 --device cpu --gui --seed 20261103 --loops 1

# 五场景CPU评估。
python -m pole_task.evaluate --checkpoint "版本记录/V1/best.pt" \
  --device cpu --seed 20261103 --output outputs/V1-evaluation/evaluation.json

# 主线默认采用B采样；先检查hardware.ready，再自行启动新训练。
python -m pole_task.training --preflight-only
python -u -m pole_task.training --max-updates 2500 --run-name V1-new-run
```

新训练仍从登记源warmup开始，执行2000次冻结Actor的warmup和500次Actor更新。`--run-name`指定目录名，同名拒绝覆盖。新训练不保证选出与V1相同的权重；V1经历1000、2200两次恢复，完整链路见版本说明。

**恢复旧模型必须使用原配置。** V1权重对应 `experiments/reward-sampling/B.json`（本地归档副本为 `版本记录/V1/training-config.json`）；此前主线均匀采样配置保存在 `experiments/reward-sampling/pre-V1-main.json`。不要直接用新版默认配置恢复历史checkpoint。

## 四、实验与对比

本版目标是改善左转连续支撑和重新承重，同时保住其他场景。A/B均从登记源warmup初始化，不从旧2100微调。

|实验|Actor起点采样|滑移权重|训练过程|
|---|---|---:|---|
|A|均匀随机|0.05|到1000恢复至2200，随后续至2500|
|B|四段轮换、段内随机|0.05|同样在1000和2200恢复，最终至2500|
|C候选|同B|0.075／0.10|离线筛选未通过，没有启动训练|

共同参数：种子20261103；Actor基础学习率1e-5，世界模型／Critic为1e-4；想象长度32，模型／策略batch均512；每次模型10批，Critic 5 epochs×4 minibatches。nominal每批4回合，straight／left_turn／right_turn各2回合。手臂权重0.05、落脚权重0.02保持不变。

B的四段起点覆盖已逐个核验；左转四个困难时段的有效候选窗口空缺从A的4/32减至B的1/32。但B实际左转策略样本数减少6.6%，因此“覆盖更均匀”不等于“左转训练量更多”。

## 五、为何选择2400

下面均为相同CPU入口、seed=20261103的五场景评估，不混用GPU训练内指标。连续承重指标剔除重新承重前10ms；触地／离地窗口40ms。正式验收仍使用完整接触统计，不用新分组放宽标准。

|模型|左转连续均速 m/s|左转再承重积分 m|五场景正式通过|
|---|---:|---:|---:|
|原footflip-2100|0.02270|0.4693|4/5|
|A-2200|0.01765|0.2626|3/5|
|B-2200|0.01832|0.2530|2/5|
|**V1 / B-2400**|**0.01434**|**0.2160**|**5/5**|
|B-2500|0.01773|0.3347|2/5|

B-2200比A-2200改善了直行等场景，但左转触地与轨迹更差；延长训练后2400取得了更均衡的结果。best.pt实际记录为2400，与编号检查点三个网络逐张量一致。2325、2425也曾通过训练内门槛，但选模分数没有超过最终2400。

V1相对B-2200：左转触地积分下降34.9%，连续均速下降21.7%，再承重积分下降14.6%，路线RMSE从2.74降至1.66cm。相对A-2200，触地积分基本相同，触地均速仍高11.3%，不能称所有落脚指标都更好。

|场景|V1连续均速 m/s|V1再承重积分 m|相对原2100：连续／再承重|
|---|---:|---:|---:|
|左转|0.01434|0.2160|−36.9%／−54.0%|
|nominal|0.01373|0.2369|−19.8%／−32.3%|
|straight|0.01124|0.0655|−26.9%／−5.0%|
|右转|0.01426|0.1915|−5.6%／−7.6%|
|eval-01|0.01419|0.1946|−9.0%／−51.5%|

所有回放完成路线，无摔倒、碰杆。左转、直行承重时间比原2100减少约7%；按承重秒归一化，左转再承重积分仍低50.4%，直行却略高2.1%。右转再承重次数92→106。V1是当前综合候选，不代表全部接触分项无退化。

2400以后loss继续下降，但2500再承重积分在左转增加55%、nominal增加120%、右转增加105%。因此保留中期best，不使用最后latest作为本版模型。当前只完成单种子固定仿真，尚未做多种子、随机扰动或实机验证。

## 六、工作区

| 目录 | 内容 |
|---|---|
| `courses/pole-weaving/pole_task/` | 主线任务与默认配置 |
| `版本记录/` | 本地版本模型、配置与验证归档 |
| `docs/versions/`、`docs/assets/demos/` | 可分享的版本记录与精选演示 |
| `experiments/reward-sampling/` | A/B原始配置及历史恢复配置 |
| `logs/`、`outputs/` | 保留的训练原始记录和分析产物，不纳入Git |
| `shared/`、`3rdparty/` | 共享资源、旧适配器和SuperMimic源码 |

[训练与恢复](docs/pole-weaving.md) · [奖励与验收](docs/foot-contact-rewards.md) · [开发日志](docs/开发日志.md) · [许可证](LICENSE)

A/B各三段日志已直接放到 `logs/q1-pole-weaving/`，不再嵌套experiment-A/B。旧试验与smoke记录移至本地 `版本记录/历史训练归档/`，可恢复；checkpoint内部历史路径与原配置保持不变，迁移映射见 `版本记录/训练路径迁移.json`。
