# 绕杆项目来源维护

本仓库当前只维护 SuperMimic 绕杆任务。后续同步应逐项审查绕杆实现及其共享依赖，不再自动合并踢球或 4v4 任务分支。

## 历史整合记录

以下记录来自此前多课程仓库，仅用于追溯，不是当前同步顺序：

| 历史来源分支 | 整合前 tip | 历史 `lecture` 合并提交 |
| --- | --- | --- |
| `SuperMimic` | `fe3004d5985e1318b5a9454ad1c25b7c1294783f` | `c930638642c5ebbaf2823e7958bc36d24992f9b0` |
| `SuperMimicPoleWeave` | `fa16a960fcaea4df7f537dc71354b4d4e029fa11` | `39ae0c4f67c2b3bdb3dca9b87f5ef7a257950100` |
| `MAPPO4V4` | `646197392ac41ebf97e067c9574d4db0dcd6e8ad` | `a40091f9cda3747399556b0882184def1689f213` |

## 当前维护要求

同步前核对实际 Git 状态；源码副本没有 Git 元数据时不要重建历史。涉及第三方依赖时核对 `shared/supermimic/upstream.json`；涉及模型与路线时核对绕杆资产 manifest。保留许可证与来源记录。

引入变更后验证绕杆训练入口、恢复、评估和回放，并确认没有重新引入踢球课程、训练包或采集工具。共享场景资产和旧版绕杆依赖的保留边界见[文件归属清单](course-file-inventory.md)。
