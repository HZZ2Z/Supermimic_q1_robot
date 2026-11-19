# 基线身份

2026-11-19：root `/home/wangzihe/公共/Supermimic_q1_robot/V2`，origin `https://github.com/HZZ2Z/Supermimic_q1_robot.git`。
初始分支 LearningHumanoidArmMotion，HEAD f297cd2bd19593a645a0dee054b64e0d9e5eca95。相比 39141c64b4e2764a0fd932edbc6ee858747210e6 仅 README 增加目标表。远端 V2 同 HEAD，已建立本地跟踪 V2。
初始未跟踪 Agent.md 已保存在 evidence/Agent.original.md；未覆盖其他用户工作。子模块 SuperMimic 118338b19dcd14f768d1e7de5a48d04b78987688，无源码修改。

初始针对 source recipe/profile/checkpoint 测试：16 通过、3 失败，失败为旧 source_root 测试夹具与 asset_root 优先规则冲突。源 checkpoint 实际严格加载及 fixture 对照通过。完成 2000 时保存 warmup、恢复预期 actor 的错误已复现；1999/2001 正常。

统一使用 env_q1，设备经权限外实测为 RTX 5070 Ti Laptop GPU 12227 MiB，CUDA 最小运算通过。受限沙箱内不可见不能解释成驱动损坏。
