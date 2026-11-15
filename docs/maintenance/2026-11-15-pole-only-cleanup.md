# 2026-11-15 绕杆专用仓库清理与验证

> 后续状态：本文记录清理时的测试结果；剩余问题现已修复，完整测试 178 项通过，详见[测试修复记录](2026-11-15-test-repair.md)。

## 结果

项目入口与现行文档已收敛为 **Q1 SuperMimic Pole Weaving**。踢球课程、`q1_soccer` 训练包、专家采集/回放工具和专属数据、六个专属测试模块及踢球/4v4 演示资源已删除。

绕杆 profile 解析迁入 `q1_supermimic.profiles`，默认选择绕杆，显式拒绝 soccer。保留配置、恢复和预检查参数转发。真实预检查复用同一输出校验：输出必须在项目 logs 子目录中，原足球目录不再被保留；已用实际预检查验证 `logs/auto` 可用且不创建运行目录。

旧版绕杆实现及必要 expert 策略/Q1 配置保留。与清理前逐文件比较，271 个绕杆与共享仿真保护文件保持原字节；唯一保护范围内的源码修改是 preflight 输出路径检查。训练参数、算法、模型接口、checkpoint 格式、路线与模型资产未改动。安装发行名和 Python 包名保持兼容，项目展示名称已更新。

## 依赖与 warmup

空的 SuperMimic 目录已恢复为原锁定提交 `118338b19dcd14f768d1e7de5a48d04b78987688`，34 个登记源码哈希全部匹配。锁文件迁至 `shared/supermimic/upstream.json`。

用户提供的根目录 warmup 已核验并移入 `courses/pole-weaving/pole_task/assets/checkpoints/warmup-checkpoint_2000.pt`，SHA-256 为 `b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5`，与 manifest 一致。该权重仍受忽略提交规则保护。

## 验证

- 入口和真实预检查输出校验：23 项通过；测试先观察到旧行为失败，再验证修正。
- 临时隔离环境中的可编辑安装通过；从仓库外导入实际项目包成功，已删除的 `q1_soccer` 不可导入。
- Python 语法、现行文档本地链接、安装映射与第三方源码哈希检查通过。
- 真实只读预检查通过，包括 warmup 严格加载与 RTX 5070 Ti Laptop GPU / CUDA 12.8 / sm_120 检查。
- GPU 冒烟训练成功，仅执行 **1 次更新**，共 4780 个环境步，停止原因为 `caller_update_limit`；生成 latest.pt 和 best.pt。
- CPU 无头回放成功：nominal-01 完成 675 步，无碰杆、无摔倒，根部 XY RMSE 为 0.043785 m。
- CPU 评估中 nominal-01、eval-01、straight、left_turn、right_turn 五组均报告 success。此处是该次冒烟产物的固定条件结果，不代表训练收敛、跨种子泛化或完成 20 次重复验收。
- 独立代码审查发现并修复了底层残留的足球输出目录限制；复核无剩余新问题。

训练产物：[运行目录](../../logs/q1-pole-weaving/run-20261115-040816)；[回放结果](../../logs/q1-pole-weaving/run-20261115-040816/verification/playback/result.json)；[评估结果](../../logs/q1-pole-weaving/run-20261115-040816/verification/evaluation/result.json)。

## 已有测试问题

完整测试执行 `python -m pytest -q --continue-on-collection-errors` 的结果为 **114 passed, 4 failed, 2 errors**，不是全绿。相同 warmup 下的清理前副本也有完全相同的四项失败（96 passed, 4 failed）：

- `tests/test_pole_course.py::test_nominal_and_held_out_splits_match_registered_acceptance_steps`
- `tests/test_pole_source_checkpoint.py::test_strict_loader_rejects_a_checkpoint_with_a_different_contract`
- `tests/test_pole_source_checkpoint.py::test_strict_loader_rejects_incompatible_actor_weight_shapes`
- `tests/test_pole_source_checkpoint.py::test_strict_loader_rejects_the_source_final_actor_as_initialization`

原因分别是旧测试仍读取历史 `passed_result`/`source_root` 字段，以及两项临时 manifest 测试只更新旧 source_root、未更新当前 asset_root，导致权重路径被错误解析。未修改这些已有测试或训练逻辑以掩盖失败。

两个收集错误来自 `test_g1_assets.py` 缺少 Isaac Gym、`test_q1.py` 缺少 SciPy，清理前也存在；共享仿真测试按方案保留。

## 备份与本地证据

修改前备份位于 `/tmp/supermimic-pole-cleanup-20261115-r2x3qa9y`：`before.tar.gz` 包含原始文件，`before-sha256.json` 记录文件哈希，`original-symlinks.tar` 补充原 expert 资源软链接。备份未包含用户随后提供的 warmup；权重现位于任务登记路径。

同目录保存前后测试日志、预检查、安装、冒烟训练和回放日志。当前副本没有有效 Git 元数据，本次没有初始化 Git、创建提交或修改历史。临时备份位于 /tmp，需长期留存时应另行保存。
