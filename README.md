<div align="center">

<sub>Q1 HUMANOID · MUJOCO · SUPERMIMIC</sub>

# SuperMimic 人形机器人三杆绕行实验

**基于开源项目 https://github.com/MohismLab/master_project_q1_robot_soccer 进行开发**

优化记录与效果展示。

[查看演示](#版本演示) · [比较指标](#效果对比) · [快速启动](#快速启动) · [V1实验报告](https://github.com/HZZ2Z/Supermimic_q1_robot/tree/V1)

**当前最新版本 V1 · B-2400**　｜　**CPU固定五场景验收 5/5**

</div>

---

## 项目介绍

本项目基于 **SuperMimic世界模型框架**，在MuJoCo中训练Q1人形机器人跟随参考动作，完成三杆S形绕行。研究重点是让机器人在跟踪路线的同时，减少足底滑移、改善落脚与支撑切换。

训练结合实际仿真采样与世界模型想象；评估同时检查路线完成、轨迹跟踪、摔倒／碰杆和实际接触滑移。足球场是仿真场景，本任务不训练踢球行为。

| 关注方向 | 如何验证 |
|---|---|
| **跟踪精度** | 根部XY轨迹与同一参考帧对齐，统计坐标RMSE |
| **接触稳定性** | 分开统计触地、离地、连续承重和再承重滑移 |
| **可比较的优化** | 保留baseline和历史模型，统一CPU场景、种子与验收口径 |

> **版本管理**：不同版本按照不同brach保存

## 版本演示

<table>
  <tr>
    <th width="33%">01 · Baseline</th>
    <th width="33%">02 · Foot slip</th>
    <th width="33%">03 · V1</th>
  </tr>
  <tr>
    <td><a href="docs/assets/demos/baseline-nominal-01.mp4"><img src="docs/assets/demos/baseline-nominal-01.gif" width="300" alt="Baseline完成一次nominal-01绕杆"></a></td>
    <td><a href="docs/assets/demos/reward-footflip-20261115-2100.mp4"><img src="docs/assets/demos/reward-footflip-20261115-2100.gif" width="300" alt="Foot slip版本2100检查点完成绕杆"></a></td>
    <td><a href="docs/assets/demos/V1-B2400-nominal-01.mp4"><img src="docs/assets/demos/V1-B2400-nominal-01.gif" width="300" alt="V1的B-2400检查点完整CPU绕杆回放"></a></td>
  </tr>
  <tr>
    <td>原始模仿与路线跟踪</td>
    <td>添加了足底滑移／落脚速度约束</td>
    <td>四段采样与两批replay</td>
  </tr>
</table>

**点击gif查看完整仿真** 

## 效果对比

`nominal-01` · CPU · seed `20261103` · 同一参考动作

| 指标 ↓ | Baseline | Foot slip · 2100 | **V1 · B-2400** |
|---|---:|---:|---:|
| 全接触滑移均速（m/s） | 0.1727 | 0.0910 | **0.0833** |
| 超速承重时间比例 | 30.48% | 16.83% | **13.58%** |
| 路线XY坐标RMSE（cm） | 2.73 | 2.69 | **1.65** |

| 五场景正式验收 ↑ | Baseline | Foot slip · 2100 | **V1 · B-2400** |
|---|---:|---:|---:|
| 通过场景数 | 0/5¹ | 4/5 | **5/5** |

¹ Baseline按后来加入的滑移门槛复评。超速定义为接触滑移速度大于0.05m/s；正式门槛保持滑移均速≤0.10m/s、超速承重比例≤20%，并要求路线正常。

**左转也有改善**：V1相对原2100，连续承重均速下降36.9%、再承重积分下降54.0%。但左转和直行承重时间约少7%，部分接触分项仍有取舍；相对B-2200，直行再承重积分增加13.6%。

[原2100数据](docs/versions/reward_footflip.json) · [V1五场景数据](docs/versions/V1-evaluation.json) · [模型身份与哈希](docs/versions/V1.json)

> 这些结果来自单种子固定仿真。5/5不是实机安全或多种子鲁棒性的证明；各项收益也不能全部归因于单一奖励或采样改动。

## 版本记录

| 版本 | 核心改动 | 分支与检查点 |
|---|---|---|
| **Baseline** | 原始模仿与路线跟踪 | https://github.com/MohismLab/master_project_q1_robot_soccer |
| **Foot slip** | 修复摩擦随机化；加入滑移／近地落脚速度惩罚；接入滑移评估和选模 | branch：`reward_footflip`，2100.pt|
| **V1** | Actor起点四段轮换；保留最近两批replay；完善采样与接触诊断，修复恢复边界和patience计数 | branch:`V1`，B-2400.pt |

V1奖励权重、学习率和正式门槛保持不变。A/B采样实验、延长训练的收益与退化，以及选择2400的依据，见[V1完整报告](https://github.com/HZZ2Z/Supermimic_q1_robot/blob/V1/README.md)。

## 快速启动

### 1 · 获取可运行源码

`main`不包含训练程序，请克隆V1分支。以下命令适用于Linux，基础环境为Conda、Python 3.11；回放支持CPU。

```bash
git clone --branch V1 --single-branch \
  https://github.com/HZZ2Z/Supermimic_q1_robot.git Supermimic_q1_robot-V1
cd Supermimic_q1_robot-V1

conda env create -f environment.yml
conda activate env_q1
python -m pip install --no-deps -e .
```

已有可用的`env_q1`时，跳过创建环境，激活后安装当前项目即可。完整依赖说明见[环境指南](https://github.com/HZZ2Z/Supermimic_q1_robot/blob/V1/docs/setup.md)。

### 2 · 准备V1权重并回放

**权重不随Git分发。** 将已取得的V1归档`best.pt`放到下方路径；`/path/to/V1/best.pt`需替换为实际文件。SHA-256必须与[版本记录](docs/versions/V1.json)一致。

```bash
mkdir -p checkpoints
cp /path/to/V1/best.pt checkpoints/V1-best.pt
sha256sum checkpoints/V1-best.pt

# 预期SHA-256：
# 575cc1fd81cf56f1258e61b4e7b2bddc05681922541146a7051a13442b042c05

python -m pole_task.play --checkpoint checkpoints/V1-best.pt \
  --case nominal-01 --device cpu --gui --seed 20261103 --loops 1
```

没有图形桌面时去掉`--gui`。回放和评估不需要训练用的源warmup权重。

### 3 · 统一CPU评估

```bash
python -m pole_task.evaluate --checkpoint checkpoints/V1-best.pt \
  --device cpu --seed 20261103 --output outputs/V1-evaluation/evaluation.json
```

<details>
<summary><strong>需要自己训练？展开查看入口</strong></summary>

先按[资产与训练说明](https://github.com/HZZ2Z/Supermimic_q1_robot/blob/V1/docs/pole-weaving.md)准备登记的源warmup checkpoint，并确认GPU可用。它与回放用的V1权重不是同一个文件。

```bash
python -m pole_task.training --preflight-only
# 确认hardware.ready为true，再执行：
python -u -m pole_task.training --max-updates 2500 --run-name V1-new-run
```

默认配置采用B采样方式；执行2000次冻结Actor的warmup和500次Actor更新。同名run拒绝覆盖。新训练不保证复现同一权重，历史模型续训必须使用其原配置。

</details>

---

<div align="center">

[当前源码 · V1](https://github.com/HZZ2Z/Supermimic_q1_robot/tree/V1)　·　[历史实现 · Foot slip](https://github.com/HZZ2Z/Supermimic_q1_robot/tree/reward_footflip)　·　[演示来源](docs/versions/demos.json)　·　[许可证](LICENSE)

<sub>保留历史对照，用相同口径验证每一次改进。</sub>

</div>
