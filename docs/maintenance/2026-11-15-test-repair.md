# 2026-11-15 测试失败与依赖问题修复

## 最终结果

在根 README 指定的 `env_q1`（Python 3.11）环境完成依赖安装与可编辑安装后，从仓库根目录执行完整测试：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q
```

结果：**178 passed in 35.49s**，0 失败、0 收集错误、0 跳过；包括真实 CUDA/Warp 测试。独立运行的绕杆、训练入口和视频测试为 **118 passed in 15.28s**。

`python -m pip check` 报告 `No broken requirements found`；`python -m q1_supermimic.train --check` 的 `training_ready` 和 GPU `hardware.ready` 均为 true，warmup 哈希及严格加载通过。

## 具体错误原因与处理

| 原问题 | 根因 | 修复 |
| --- | --- | --- |
| `test_nominal_and_held_out_splits_match_registered_acceptance_steps`：`KeyError: passed_result` | 历史验收结果已标记为 provenance-only，不再作为必须随仓库分发的运行资产返回；测试仍尝试读取原始文件 | 直接验证随仓库分发的路线/保持动作参考，保留 675、671、198、298、298 个控制步及 train/held_out 划分断言，不再依赖未分发的历史结果文件 |
| `test_strict_loader_rejects_the_source_final_actor_as_initialization`：`KeyError: source_root` | 清单已使用 asset_root，测试仍读取旧字段 | 复制清单时将 asset_root 指向真实仓库根目录，继续断言最终 Actor 不能用作训练初始化 |
| `test_strict_loader_rejects_incompatible_actor_weight_shapes` 和 `test_strict_loader_rejects_a_checkpoint_with_a_different_contract` 报错与预期不符 | 临时清单只修改旧 source_root，却继承了实际优先使用的相对 asset_root，导致错误访问 `/tmp/warmup.pt`，尚未进入模型校验即失败 | 临时清单设置正确的 asset_root；保留错误张量尺寸及 goal_representation 的精确拒绝断言，现在真正执行到相应检查 |
| G1 测试收集失败：缺少 isaacgym | 测试只使用几何函数，gymapi 导入未使用，也未启动 Isaac Gym 后端 | 删除未使用导入，使用仓库已有独立四元数函数；保留 Pinocchio 与 MuJoCo 的独立运动学对照，未跳过或删除测试 |
| Q1 测试收集失败：缺少 scipy；G1 随后暴露缺少 pinocchio | 基础训练依赖未包含共享资产测试使用的独立数值参考库 | 新增 requirements-test.txt，复用基础 requirements.txt，补充 scipy==1.17.1 与 pin==4.1.0，并安装到现有 env_q1 |

恢复收集后，还修正了原共享测试遗留的根目录相对路径：G1 以测试文件位置解析共享 YAML 和资源的绝对路径；Q1 使用自动恢复工作目录的 monkeypatch fixture，将这些旧工具限定在 shared/q1-mujoco 下运行。没有恢复根目录软链接。

替换 G1 四元数工具后，已有测试发现 Pinocchio 的 float64 与相机配置默认 float32 混用；将对照计算的两个输入统一为 float64，保持原精度阈值和比较内容。未修改生产数学函数。

## 按 README 复现

继续使用根 README 的 Python 3.11/Conda 安装路线：

```bash
conda activate env_q1
python -m pip install -r requirements-test.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest
python -m q1_supermimic.train --check
```

上游子模块 README 的 Python 3.13/uv 指令面向其独立 Q1 V2 项目，不与本项目绕杆适配环境混装。基础 requirements.txt 未修改，固定训练依赖版本不变。

## 修改边界与证据

本次只修改四个测试模块，新增测试依赖清单并补充环境指南；没有修改绕杆训练算法、模型接口、checkpoint 格式或资产，也没有删除、跳过测试或放宽验收阈值。独立复核无待处理问题。保留了本次开始时用户已有的根 README 内容。

本次修改前测试与文档备份、依赖解析/安装日志、修复前后测试日志和预检查输出保存在 `/tmp/pole-test-repair-20261115-b0utiohx/`。该目录为本地临时证据；需要长期保存时请另行归档。
