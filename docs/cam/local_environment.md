# 本地环境

完整只读核验记录：`evidence/p0-preflight.json`。Ubuntu 24.04.5，Python 为 `/home/wangzihe/miniconda3/envs/env_q1/bin/python`。未新建环境或升级依赖。

Torch 2.9.0+cu128、MuJoCo 3.17.0、numpy 2.4.6。实际设备 NVIDIA GeForce RTX 5070 Ti Laptop GPU，12227 MiB，sm_120 在 Torch 编译架构列表中；权限外最小 CUDA 运算通过。沙箱中设备不可见，需在支持 GPU 设备访问的执行上下文运行。

源 warmup checkpoint SHA256 `b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5`；严格加载通过。XML、参考、manifest、源 TOML 哈希核验通过。

P0 资源快照见 JSON：主存约 14 GiB、可用约 9 GiB，磁盘可用约 139 GiB。正式资源档位必须以三组完整更新实测确定，不由静态显存推断。
