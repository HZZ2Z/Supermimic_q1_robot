# Pole Weaving Project Entry

## Purpose

为 Q1 SuperMimic Pole Weaving 提供单一任务入口，帮助使用者配置环境、训练、恢复、评估和回放绕杆策略。

## Requirements

### Requirement: 项目入口聚焦绕杆
项目 SHALL 在中英文主页提供绕杆任务说明和安装入口，不再提供踢球或 4v4 课程导航。

#### Scenario: 打开项目主页
- **WHEN** 使用者打开 README
- **THEN** 能够直接进入绕杆任务，并找到运行所需的环境、warmup 和 checkpoint 说明

### Requirement: 提供完整运行指引
项目 SHALL 提供预检查、一次更新的冒烟训练、正式训练、恢复、无头评估、验收和回放命令，并说明硬件与外部权重要求。

#### Scenario: 检查首次运行条件
- **WHEN** 使用者按说明运行只读预检查
- **THEN** 系统校验登记资产与硬件，缺少权重时明确失败且不创建训练目录

#### Scenario: 冒烟训练
- **WHEN** 固定版本依赖、warmup 权重和硬件均满足要求并执行一次更新
- **THEN** 生成可加载的绕杆 checkpoint，但不将管线通过解释为策略达到任务指标

### Requirement: 保留绕杆兼容边界
项目 SHALL 保留旧版绕杆及其必要策略和配置依赖，保留共享机器人和场景资源、资产来源及许可证。

#### Scenario: 阅读维护清单
- **WHEN** 维护者查看仍带有足球命名的资产
- **THEN** 文档解释其绕杆复用或历史来源用途，不将历史记录作为现行踢球任务入口
