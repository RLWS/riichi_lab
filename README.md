# RiichiLab 接入项目

独立四麻适配层，计划连接 RiichiEnv 与 train_mj 推理服务，用于本地对局回归及 riichi.dev 在线机器人。

当前已完成 A0 与 A1：probe、配置校验、每席状态与动作适配、两种 profile、HTTP 后端注入和手工四席单局可用。v1.2.0/v1.3.0 真实双服务已联调；完整 A2 parity、自动编排与强度统计尚待实现。

- [可行性与问题审查](docs/feasibility-review.md)：审查结论、问题清单、上游引擎实测及验证边界。
- [修订接入方案](docs/riichi-dev-integration.md)：状态/动作契约、评测方法、在线生命周期和分阶段验收。
- [开发计划](docs/development-plan.md)：A0–B2 里程碑任务分解、退出检查与测试矩阵。
- [兼容性记录](docs/compatibility.md)：四类契约基线（HTTP 结构 / feature_schema / 权重与校准 / 引擎与规则）。
- [多后端配置契约](docs/multi-backend-contract.md)：v1.2.0/v1.3.0 两个实例、不同 profile、四席绑定与 A1/A3/A4 分工。
- 里程碑记录：[A0 契约探测](docs/milestones/A0.md)。
- [A1 最小闭环与运行方式](docs/milestones/A1.md)：双服务证据、历史特征口径及验收边界。

下一步是 A2 双版本完整特征一致性验证，通过后再推进 A3/A4。
