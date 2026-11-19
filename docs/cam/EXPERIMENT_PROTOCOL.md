# 当前补充协议

2026-11-20：先执行 [LR_REPLAY_RUNBOOK.md](LR_REPLAY_RUNBOOK.md) 预登记的三组 LR/replay 诊断。原 CAM 校准/晋级协议保留但暂缓；预算共用 RUN_SLOTS.json 的七槽位上限，不另起七次。

# CAM v2 首轮冻结协议

正式组为 B0/B_ref/B_task seed=0，随后仅在候选晋级时扩展 B0 与一个候选 seeds=1,2。每槽位最多一次有依据的合格重试；诊断单独记录。默认全部从登记源 warmup 初始化，项目 2000 warmup + 500 Actor，horizon32、FP32、同一资源档位。B0 不运行 CAM。候选从首个 warmup 即使用各自奖励拟合 Critic，Actor 冻结。

主指标为 nominal-01 与 eval-01 等权的 root SO(3) 角误差 RMS。改善至少5%，且大于基线重复回放数值波动。全部原任务门槛、5个登记用例和20次 nominal 验收须通过。所有用例的足滑、action 一阶时间差分 RMS、32步模型状态及 CAM 误差恶化不得超过5%或已登记数值噪声（取较大者）。具体计算在 cam_evaluation.PROTOCOL。两候选通过按主指标、足滑、动作平滑依次排序，完全相同时选 B_ref。固定 eval seed=20261103。

root XY RMSE 和 best 选择沿用原实现。eval-01 为选模验证用例，不是独立泛化测试。nominal 20次固定回放不代表20种扰动。多种子逐项报告，不自动声称统计显著性。

B0 返回后，仅用训练参考和 B0 训练用例轨迹做一次固定离线校准。H0=参考竖直CAM RMS，下限1e-6；sigma=两目标归一化误差平方中位数/ln2，下限1e-4。共同 lambda 取两候选允许值的最小值，同时限制32步想象奖励和动作梯度 RMS 为原奖励量级的10%。不做优化器更新，不用 eval-01。保存样本和计算证据后冻结候选配置，复核完整 GPU 路径。

目前 configs/cam 的候选尺度/权重是非零诊断值，未校准前不得启动正式候选。先测三组完整更新，再确定512/256/128共同档位。128仍不可行标记 BLOCKED_RESOURCE。

源码、依赖和语义修复发生在正式运行前；正式运行后如改训练语义，必须新实验版本，不能混用旧对照或以重试名义扩张预算。所有正式训练通过 scripts/cam_local.py 启动并记录 PID、命令、日志、退出码与 checkpoint，W&B 离线。

测量细节：足滑仅统计 foot geom 与 floor plane 的 dist<=0 接触，使用接触世界坐标处的平动 Jacobian 乘 qvel，去掉法向分量后对所有接触样本求速度 RMS；不把足 link 原点速度当接触点速度。action 平滑包含初始 previous action 到第一个 action 的差分，dt=.02。模型状态总RMSE使用原始57物理列并对齐四元数符号，因混合单位，仅作同定义对照，同时报告位置/关节/速度分项。所有32步固定动作窗口步距32，失败短轨迹若无完整窗口则指标缺失并拒绝候选晋级。
