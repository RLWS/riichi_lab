# RiichiLab 接入项目

独立四麻适配层，计划连接 RiichiEnv 与 train_mj 推理服务，用于本地对局回归及 riichi.dev 在线机器人。

当前已完成 A0/A1、A2 检查工具、A3 工程验收与 A4 正式强度验收。正式实验为 128 个独立种子、1,792 半庄，全部产物及 1,102,487 次决策离线回放通过；按预登记规则，v1.3.0 相对 v1.2.0 的强度结论为证据不足。原冻结 tag 的 DEF-001 保留为已知失败，上游修复已在新分支提交并验证。

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

- [A4 正式强度验收](docs/milestones/A4-formal.md)：完整资格审计、正式指标、阈值判定和证据。

- [手动启动强度对局](docs/run-strength.md)：128 个独立种子、1,792 半庄，开局前登记协议，单轮运行命令。

- [B0 在线协议模拟验收](docs/milestones/B0.md)：异步客户端、动态预算、ack 确认、合法备用及模拟服务器测试。

A4 与 B0 已完成，不自动追加强度样本或调整模型。真实平台验证 B1、有界排位 B2 及训练数据/权重修复采纳另行安排。
