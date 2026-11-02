# 可行性与设计审查

审查日期：2026-11-02。对象：本仓库原接入方案、相邻 train_mj 的真实接口/编码/采样代码、RiichiEnv 源码及发行包、riichi.dev 官方协议。修订后的实施设计见 [接入方案](riichi-dev-integration.md)。

## 1. 判断

**值得继续，但应先做正确性闭环，暂不把方案视为可直接上线的设计。**

- 工程可行：已有引擎、合法动作、可见事件和完整状态推理 API，无需自研计分或训练新模型即可开始适配。
- 主要成本：训练状态语义的重建、不同决策时点的快照、复合动作与真实引擎动作对齐；不是 HTTP 或 WebSocket 收发。
- 线上有条件可行：公开协议具备所需信息，但未验证真实服务器序列化、规则及网络兼容性。
- 棋力收益未知：当前模型是候选排序模型，离线指标与接入能力不足以推出实战强度；需固定基线与统计实验。
- 仓库成熟度：审查开始时只有一份设计文档，没有 Python 包、依赖配置、测试或可运行 CLI。本文实测的是上游引擎能力，不是本项目适配成功。

## 2. 原设计问题与修正

P0 表示实现前必须消除的正确性问题；P1 表示批量评测或上线前必须解决的问题；P2 表示范围与维护优化。

| 级别 | 原设计问题 | 后果与优化 |
|---|---|---|
| P0 | 在线散发事件先 tracker.apply，随后 act 又消费 obs.new_events | 重复摸牌、弃牌、副露；统一只消费 Observation，外层只记传输日志 |
| P0 | riichi_discard 直接回复 reach+dahai | 训练为复合动作，引擎为两阶段；增加候选展开与 PendingRiichiPlan，每请求只发一次 |
| P0 | 忽略 reach_accepted，在宣言弃牌后自行生成 reach2 | 宣言被荣和等情况会错误扣款、标记立直；以接受事件及权威状态确认 |
| P0 | remaining_draws_estimate 被写成初始 70 的计数 | HTTP 必需 remaining_tiles；服务端派生 estimate=remaining_tiles/4。补齐 round_number、match_rounds、own_melds 等必需字段 |
| P0 | tsumo 统一推进 turn_number | 实际训练字段是自家摸牌次数；只增加对应玩家计数，不能按全桌事件累加 |
| P0 | 抢杠响应前先应用完整加杠 | 特征提前看到副露升级；保留宣杠前快照和独立目标，窗口结束后提交 |
| P0 | 把 select_action_from_mjai 当作合法性和语义兜底 | 可返回 None，也能宽松匹配错误输入；候选绑定原始 Action，模型输出严格回查 |
| P0 | 纯事件 tracker 默认能复原全部振听 | 放弃合法荣和与实际动作确认并非纯公开事件；补决策反馈，未知信息不能臆造 |
| P1 | 对局中的点数变化等到下一局覆盖 | reach 接受后的点况/供托输入错误；维护即时分数，避免增量与权威覆盖双扣 |
| P1 | 向预测 API 请求服务端特征逐位对拍 | 现有响应不含向量；开发/CI 使用固定版本本地编码器，HTTP 单独测契约 |
| P1 | 要求所有特征无条件一致 | 上游已记录红五加杠目标差异；登记已知缺陷，保持真实合法牌，不静默忽略 |
| P1 | deadline 固定 18 秒，自动重连恢复 | 时间库逐请求衰减且原局不能恢复；动态预算、异步推理、ack 确认和断线生命周期 |
| P1 | 100 半庄/四同模型自对弈当作强度证据 | 只能说明稳定性；增加固定对手、轮换配对、按种子成簇的置信区间及预定停止条件 |
| P1 | API 路径/种子被当作充分复现记录 | 保存权重哈希、规则、commit、dirty、依赖锁和实际校准；同 seed 不保证策略分叉后同牌山 |
| P2 | 首版同时实现 HTTP 与库推理、任意版本依赖及本机路径 | 扩大维护面且不可移植；生产仅 HTTP，库依赖只用于对拍，精确锁定版本 |
| P2 | 全量决策快照长期无限写盘，错误即整批丢失 | 有界异步日志、压缩分片、错误局完整保存；评测失效但保留已产出证据 |

## 3. 已执行的实测

使用当前 Windows、Python 3.11.5，通过 uv 临时环境安装 riichienv==0.4.11，未把它加入本仓库生产依赖。两次都使用 game_mode=4p-red-half、GameRule.default_tenhou()、seed=42。

| 探测 | 结果 |
|---|---|
| 随机合法动作半庄 | 完成 1056 次 env.step；点数 [25000,25000,25000,25000]，顺位 [1,2,3,4]；未触发立直，不能作为立直覆盖 |
| 简单向听数弃牌策略半庄 | 完成 833 次 env.step；点数 [22000,28800,38700,10500]，顺位 [3,2,1,4]；捕获和牌及立直 |
| 立直动作形式 | 合法 Riichi 的 tile=None、consume_tiles=[]，不包含模型应选弃牌 |
| 立直辅助函数 | check_riichi_candidates 返回 136 牌 ID；真实观察中曾返回 [90,113]，而不是所有弃牌 |
| 事件顺序 | 实际捕获 tsumo → reach → dahai → reach_accepted → 下一家 tsumo |
| Observation 序列化 | serialize_to_base64 后 deserialize_from_base64，合法动作数量一致；这是本地往返，不是服务器兼容证明 |
| 事件数据类型 | new_events 返回 JSON 字符串，需要解析 |
| 宽松匹配 | select_action_from_mjai({"type":"dahai"}) 返回非空；缺牌面仍匹配，不能单独作为正确性检查 |

以上两场仅验证引擎能力；没有连接在线端点，没有运行模型权重，没有完成 tracker/parity，也没有宣称项目测试通过。随机及简单策略结果不能用于模型强度评价。原方案引用的“百万半庄”是上游 Mortal 兼容性报告，不是本项目保证。

## 4. 证据定位

本地 train_mj 基线 commit：1e5092f344fb69bcdf4697f683f2ecf15b9865d8（审查只读）。主要证据路径相对于该仓库：

- docs/full-state-api.md：字段单位、完整 state、抢杠快照及已知红五边界。
- mjai/serving/validation.py：_validate_full_state、remaining_tiles/4 派生及候选限制。
- mjai/api/schemas.py、mjai/api/routes.py：真实请求/响应，无特征向量端点；no_yaku_call_penalty 不在预测请求中。
- mjai/data/builders/sample_builder.py：_player_view_snapshot、自家 draw_counts、抢杠窗口前采样。
- mjai/rules/state.py：reach1/reach2、分数单位转换、鸣牌后状态与振听更新。
- mjai/features/action_features.py：六类 turn 与七类 response 候选（不含 unknown）。

外部证据：

- [RiichiEnv 固定源码基线](https://github.com/smly/RiichiEnv/tree/0ee1236a283b15508f3ac28407027f340b81e1de)：核对 Action、Observation 与包元数据。
- [动作序列化](https://github.com/smly/RiichiEnv/blob/0ee1236a283b15508f3ac28407027f340b81e1de/riichienv-core/src/action.rs)、[动作匹配](https://github.com/smly/RiichiEnv/blob/0ee1236a283b15508f3ac28407027f340b81e1de/riichienv-core/src/observation/mjai_select.rs)：两阶段立直和宽松匹配依据。
- [Mortal 迁移](https://riichi.dev/docs/mortal)：Observation 单一增量输入。
- [MJAI 协议](https://riichi.dev/docs/protocol)：请求、动态时间、ack、单条回复、断线行为。
- [验证流程](https://riichi.dev/docs/validation)：验证状态与实际通过条件。

网站信息是审查当日的公开契约，实施时仍要保存真实协议夹具并核对服务端兼容性。

## 5. 建议投入顺序

先做 A0 契约夹具，再完成 A1 单局闭环与 A2 特征对拍。这三个阶段解决最大不确定性；通过后才做大批量 A/B 和线上接入。没有适配器吞吐与真实权重延迟实测前，不给出大规模评测耗时或确定上线工期。

继续推进的前提是：可见信息足够构建完整 state；每个模型候选能映射到实际合法动作；对拍无未解释差异。任何一项无法满足，优先缩小范围或修复契约，不用增加对局数量掩盖问题。
