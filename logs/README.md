# V1训练日志

保留A/B两组各三段训练，日志直接位于q1-pole-weaving下。

|运行|阶段|最后更新|
|---|---|---:|
|A-to1000|均匀采样warmup，OOM中断|1000|
|A-from1000-to2200|恢复warmup并训练Actor|2200|
|A2200_2600|计划2600，实际提前停止|2500|
|B-stratified-to1000|warmup|1000|
|B-stratified-from1000-to2200|恢复warmup，Actor四段采样|2200|
|B2200_2500|Actor续训，best为2400|2500|

metrics.jsonl为训练内评估与loss；collection.jsonl为回合起止帧和有效窗口。二者保持原始内容。config为运行快照，identity中的个人主目录和外部源绝对路径已替换，原哈希保留；不要把脱敏identity用于恢复校验。路径迁移不改写历史来源编号。中断运行缺少stop_reason.json是原始状态，不补造结束记录。

模型权重、MuJoCo XML、参考动作缓存不上传。日志中的Infinity表示尚无合格best_score；分组指标null通常表示该分组没有样本，不等于训练NaN。CPU复评见docs/versions/V1-evaluation.json，不与训练GPU结果混用。
