# RiichiLab 接入项目

独立四麻适配层，计划连接 RiichiEnv 与 train_mj 推理服务，用于本地对局回归及 riichi.dev 在线机器人。

当前已完成 A0/A1、A2 检查工具、A3 工程验收与 A4 统计工具验收：两版本真实双服务完成 112 半庄并完整重跑，逐局结果与输入/动作摘要一致，正式两轮零非法动作/断言/fallback。A4 支持配对种子簇 bootstrap 和辅助指标；现有 A3 数据的探索性结论为证据不足，正式强度实验尚未启动。原冻结 tag 的 DEF-001 保留为已知失败，上游修复已在新分支提交并验证。

- [可行性与问题审查](docs/feasibility-review.md)：审查结论、问题清单、上游引擎实测及验证边界。
- [修订接入方案](docs/riichi-dev-integration.md)：状态/动作契约、评测方法、在线生命周期和分阶段验收。
- [开发计划](docs/development-plan.md)：A0–B2 里程碑任务分解、退出检查与测试矩阵。
- [兼容性记录](docs/compatibility.md)：四类契约基线（HTTP 结构 / feature_schema / 权重与校准 / 引擎与规则）。
- [多后端配置契约](docs/multi-backend-contract.md)：v1.2.0/v1.3.0 两个实例、不同 profile、四席绑定与 A1/A3/A4 分工。
- 里程碑记录：[A0 契约探测](docs/milestones/A0.md)。
- [A1 最小闭环与运行方式](docs/milestones/A1.md)：双服务证据、历史特征口径及验收边界。
- [A2 对拍与上游修复](docs/milestones/A2.md)：普通 XML 通过结果、DEF-001 复现和新分支验证。

- [A3 批量运行与验收](docs/milestones/A3.md)：固定种子、席位轮换、故障处理及重跑一致性。

- [A4 统计工具与探索性报告](docs/milestones/A4.md)：计数口径、区间、复现检查及正式实验边界。

- [手动启动强度对局](docs/run-strength.md)：128 个独立种子、1,792 半庄，开局前登记协议，单轮运行命令。

正式强度运行入口和预设协议已准备，等待用户手动执行；完成后再核验登记与产物并做正式统计。训练数据与权重的修复采纳单独安排。
