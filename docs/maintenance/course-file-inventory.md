# 绕杆项目文件归属与入口

仓库专门维护 Q1 SuperMimic Pole Weaving。安装方式见[环境指南](../setup.md)。现有目录和 Python 包名保持不变。

| 规范位置 | 当前用途 |
| --- | --- |
| `courses/pole-weaving/pole_task/` | 新旧绕杆实现、场景、训练、评估、回放、配置与路线数据 |
| `courses/pole-weaving/q1_supermimic/` | 绕杆 profile 解析与便捷训练入口 |
| `shared/supermimic/expert/` | 旧版绕杆使用的 `ActorCritic`、Q1 配置与来源/许可证 |
| `shared/supermimic/upstream.json` | SuperMimic 上游来源版本及原始文件哈希（不代表本地修改后哈希） |
| `shared/q1-mujoco/` | 共享仿真基础设施、模型、纹理、动作和工具 |
| `3rdparty/SuperMimic/` | 上游来源快照及本地奖励修改（普通目录） |
| `scripts/` | 机器人模型生成和动作重定向工具 |

## 训练接口

默认运行 `python -m q1_supermimic.train`，等价于选择 `pole-weaving`。保留 `--profile pole-weaving`、`--config`、`--resume` 和 `--check`；后两项分别转发为绕杆命令的 `--resume-checkpoint` 和 `--preflight-only`。输出必须位于仓库 `logs/` 内，默认 `logs/q1-pole-weaving/`。

完整训练接口仍为 `python -m pole_task.training`，评估、验收和回放分别使用 `pole_task.evaluate`、`pole_task.acceptance`、`pole_task.play`。绕杆 checkpoint 格式、模型维度和配置路径不变。

## 保留的兼容内容

旧版绕杆训练辅助函数、模型适配器、环境和测试保留。`expert.policy` 与原 Q1 配置不改名、不改参数，以保护其加载与控制行为。足球场 XML、球门和纹理被绕杆场景复用，因此保留原文件名；球和球门不参与绕杆奖励或终止条件。

共享仿真基础设施和维护脚本保留，不能按文件名中的 `soccer` 或 `kick` 批量删除。这些资源不代表项目仍提供踢球模仿学习任务。

`setup.py` 映射源码目录；发行名 `q1-robot-soccer` 和现有可编辑安装路径文件名保留用于安装兼容。`tests/conftest.py` 仅为测试加入源码路径。

## 已移除与历史记录

踢球课程、`q1_soccer` 训练包、足球专家采集/回放工具及专属数据和测试已移除，没有兼容转发入口。原 `q1_soccer.profiles` 的绕杆解析功能现在由 `q1_supermimic.profiles` 提供。

根目录与资产目录的 `SOURCE.json`、`expert/SOURCE.json` 以及 OpenSpec 归档保留为历史来源记录；其中旧路径不作为当前文件存在性保证。当前任务指引以绕杆 README、配置和资产 manifest 为准。
