# RiichiLab 接入项目

独立四麻适配层，计划连接 RiichiEnv 与 train_mj 推理服务，用于本地对局回归及 riichi.dev 在线机器人。

当前处于设计阶段，尚无可运行的适配器或 CLI。

- [可行性与问题审查](docs/feasibility-review.md)：审查结论、问题清单、上游引擎实测及验证边界。
- [修订接入方案](docs/riichi-dev-integration.md)：状态/动作契约、评测方法、在线生命周期和分阶段验收。
- [开发计划](docs/development-plan.md)：A0–B2 里程碑任务分解、退出检查与测试矩阵。
- [兼容性记录](docs/compatibility.md)：四类契约基线（HTTP 结构 / feature_schema / 权重与校准 / 引擎与规则）。
- 里程碑记录：[A0 契约探测](docs/milestones/A0.md)。

实施从 A0 契约探测开始，优先完成最小本地闭环和特征一致性验证。
