# RiichiLab 接入项目

独立四麻适配层，计划连接 RiichiEnv 与 train_mj 推理服务，用于本地对局回归及 riichi.dev 在线机器人。

当前已完成 A0/A1 和 A2 检查工具：两版本真实双服务单局、XML/状态/候选特征对拍及 API 契约检查可用。原冻结 tag 因已知红五抢杠缺陷仍未通过 A2；上游修复已在新分支提交并验证。自动编排与强度统计尚待实现。

- [可行性与问题审查](docs/feasibility-review.md)：审查结论、问题清单、上游引擎实测及验证边界。
- [修订接入方案](docs/riichi-dev-integration.md)：状态/动作契约、评测方法、在线生命周期和分阶段验收。
- [开发计划](docs/development-plan.md)：A0–B2 里程碑任务分解、退出检查与测试矩阵。
- [兼容性记录](docs/compatibility.md)：四类契约基线（HTTP 结构 / feature_schema / 权重与校准 / 引擎与规则）。
- [多后端配置契约](docs/multi-backend-contract.md)：v1.2.0/v1.3.0 两个实例、不同 profile、四席绑定与 A1/A3/A4 分工。
- 里程碑记录：[A0 契约探测](docs/milestones/A0.md)。
- [A1 最小闭环与运行方式](docs/milestones/A1.md)：双服务证据、历史特征口径及验收边界。
- [A2 对拍与上游修复](docs/milestones/A2.md)：普通 XML 通过结果、DEF-001 复现和新分支验证。

下一步明确上游修复的版本/数据采纳并重跑 A2 严格闸门，再推进 A3/A4。
