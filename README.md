# reward_footflip · 足底稳定性改进

Q1 人形机器人基于 SuperMimic 的三杆绕行实验。项目总览和各版本演示见 [main](https://github.com/HZZ2Z/Supermimic_q1_robot/tree/main)。分支名沿用 `reward_footflip`，实际新增项为 foot slip 与近地落脚速度惩罚。

## 本版改动

- 修复摩擦随机化被地面优先级覆盖的问题。
- 沿用原有辅助奖励入口，加入足底滑移（0.05）和近地向下落脚速度（0.02）惩罚。
- Actor 每 25 次更新重新采样，每 50 次保存中期 checkpoint。
- 记录分项奖励、采样量和实际接触滑移；选模同时检查路线和滑移。

## 展示结果

展示 `run-20261115-160417/checkpoint-002100.pt`，不是最终模型。与旧 baseline 比较，同为 `nominal-01`、CPU、固定初始条件：

| 指标 | Baseline | 2100 |
|---|---:|---:|
| 滑移均速 | 0.1727 m/s | **0.0910 m/s（降低 47.3%）** |
| 超过 0.05 m/s 的承重时间比例 | 30.48% | **16.83%** |
| 根部 XY RMSE | 2.73 cm | **2.69 cm** |
| 路线完成／摔倒／碰杆 | 完成／无／无 | 完成／无／无 |

[查看独立视频对比](https://github.com/HZZ2Z/Supermimic_q1_robot#版本演示) · [指标与版本记录](docs/versions/reward_footflip.json)

2100 的五场景 CPU 评估通过 4/5，左转仍未通过实验滑移门槛，因此没有合格的 `best.pt`。结果反映整组改动，未单独归因于某一奖励；尚未进行实机验证。手臂姿态和评估门槛仅作分析，本版未据此改动。

## 使用

```bash
git clone --branch reward_footflip https://github.com/HZZ2Z/Supermimic_q1_robot.git
cd Supermimic_q1_robot
conda env create -f environment.yml
conda activate env_q1
python -m pip install --no-deps -e .
python -m pole_task.training --preflight-only
```

训练前按[环境指南](docs/setup.md)与[资产说明](docs/pole-weaving.md)准备 warmup 权重。模型权重、原始录像和运行日志不随 Git 分发。第三方源码已随分支保存，包含本地修改，无需初始化子模块。

详细说明：[奖励与采样](docs/foot-contact-rewards.md) · [训练／评估／回放](courses/pole-weaving/README.zh-CN.md) · [开发日志](docs/开发日志.md) · [许可证](LICENSE)
