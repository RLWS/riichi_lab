# 兼容性记录（A0-5 契约登记）

> 创建时间：2026-11-02。按方案 §3 将兼容性分开为四类记录；本文件是当前基线快照。
> 基线证据：`tests/fixtures/http_contract/`（黄金请求与响应）、`tests/fixtures/engine/index.json`（引擎夹具）。
> 升级流程：任一类别变化 -> 重跑 A0-4 契约探测与 A2 对拍（开发计划 §7），再决定是否采纳。

## 1. HTTP 结构

- 服务：train_mj serve-api。修复后的黄金请求使用真实 v1.3.0 权重经 FastAPI TestClient 联调（CPU、无监听端口），验证记录为 tests/fixtures/http_contract/verification.json；这证明 HTTP 路由契约，不代替 TCP/在线网络验收。
- 端点：`GET /health`、`GET /v1/models`、`POST /v1/predict/turn`、`POST /v1/predict/response`。
- 请求：`state` + `legal_actions` + `top_k`；response 追加可选 `pass_bias`（`null`/缺省 = 服务端默认，显式 `0` = 覆盖）。
- 响应字段：`task`、`selection_mode`（turn=raw；response=calibrated）、`selected_action`、`predictions[]`（`rank/index/action/score/prob`）、response 另有 `pass_bias`、`no_yaku_call_penalty`、`raw_selected_action`；`model` 块见下。
- **响应不返回编码特征向量**（对拍在开发进程用固定版本编码器完成，方案 §9）。
- 候选上限：**turn `candidate_limit=64`，response `candidate_limit=16`**（候选绑定超过即明确失败，方案 §5；引擎 response 窗口一般不超过 16，仍以登记值为准）。
- 错误语义（train_mj serving-api 约定）：400 结构错误 / 401 缺 key / 413 超候选上限 / 422 state 校验失败 / 503 未就绪。

## 2. feature_schema 语义

- 版本：`full_state_v2`；缺失或版本不符 -> 422，不会静默回退。
- 维度：turn state/action = **11969 / 89**；response = **12012 / 91**。
- 必需字段以锁定 tag 版本的 `mjai/serving/validation.py::_validate_full_state` 为准（26 个：feature_schema、actor、dealer、seat_wind、round_wind、round_number、match_rounds、remaining_tiles、all_melds、own_melds、scores、discard_history、public_events、riichi_declared_by_player、riichi_turn_by_player、all_discards、dora_indicators、hand、honba、riichi_sticks、turn_number、riichi_declared、temporary_furiten、riichi_furiten、last_discard_actor、last_discard_tile，response 另需 response_target）。本节仅指 full_state_v2；旧版见多后端契约。
- 派生量：`remaining_draws_estimate = remaining_tiles / 4`（服务端计算，不是请求字段）。
- 座位：绝对座位入参，`seat_wind=(actor-dealer)%4` 强校验，服务端统一旋转相对视角。
- 点数为整数实际点数；`round_wind/round_number` 0..3；`match_rounds ∈ {4,8,null}`；`remaining_tiles` 0..70。

## 3. 模型权重与校准（部署清单）

A0-4 实测基线（`tests/fixtures/http_contract/baseline_models.json`）：

| 项 | turn | response |
|---|---|---|
| checkpoint | `runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt` | `.../best_response.pt` |
| SHA256 | `30f2811cd86beebad57d4da1428272b25b598fb64a6231b5a1760e11ec08bf90` | `e1d8c58a2aa61cef23852f15fe5bb9c65014bc3916c52756fb8c4cd1cd956c51` |
| checkpoint_task | multitask | multitask |
| device / hidden_size / dropout | cpu / 512 / 0.1 | cpu / 512 / 0.1 |
| candidate_limit | 64 | 16 |
| default_pass_bias | null | 1.5 |
| no_yaku_call_penalty（服务端配置，非请求参数） | 0.0 | 0.0 |

- operating point：`pass_bias=1.5 / no_yaku_call_penalty=0`（train_mj 2026-10-28 校准结论；适配层默认 `pass_bias=null` 透传服务端配置）。
- checkpoint 路径不能代替哈希；本表哈希即部署清单（方案 §6）。评测期间模型元信息变化 -> 运行失效。
- `hidden_size` 等网络参数仅记录，不作为适配层兼容性条件（方案 §3）。

## 4. 引擎版本与规则

- 引擎：**riichienv==0.4.11**（PyPI 发行包；发行包与主分支不视为相同，验收以锁定包为准）。
- 对局配置：`game_mode="4p-red-half"`、`GameRule.default_tenhou()`、种子固定。
- 运行环境：Windows / Python 3.11.5 / websockets 17.4（`additional_headers` 可用）/ httpx 0.28.1 / PyYAML 6.0.3（依赖锁：`uv.lock`）。
- 黄金夹具：tests/fixtures/engine/，修复字符串解析和字牌识别后重采集 60 局（seeds 1-60）。45,720 个决策、458,100 条合法动作保留 MJAI 与原始物理牌；所有决策在采集时检查完整往返，3,209 份代表 Observation 入库并由测试重新解码比较事件、关键状态及完整合法动作，不再读取缓存布尔值作为证明。
- 夹具覆盖：种子对局覆盖跨局终局、立直、吃碰/三类杠、宝牌、摸切/手切；抢杠与双响由 directed/chankan.json、directed/multi_ron.json 定向引擎设置和真实 step 结果补齐，测试重建触发窗口及和牌结果。它们是局部定向场景，不声称完整半庄。

## 5. 已知缺陷登记（对拍独立失败项）

| 编号 | 内容 | 来源 | 处理 |
|---|---|---|---|
| DEF-001 | 训练采样器的抢杠目标取原碰叫牌；实际加杠牌与原叫牌红五属性不同时，目标特征存在差异 | train_mj full-state-api.md"已知训练侧边界" | 适配层始终用实际加杠牌构造候选与 response_target；parity 中该差异单列失败、阻断未经解释的上线，不伪造牌追求逐位一致 |

2026-11-02 后续：已在 codex/fix-chankan-red-target 分支提交 829c66e 修复采样器并通过测试，详见 A2 记录。当前 v1.2.0/v1.3.0 tag 和既有样本/权重未改变；默认冻结版本闸门仍应报告此缺陷，不能仅因修复分支通过就关闭原版本登记。

## 6. train_mj 版本锁定

- train_mj 以 **tag** 固定版本（该仓库每次参数量调整打新 tag）。
- 已锁定 tag：**v1.3.0**（annotated，2026-11-02 打于 commit `1e5092f344fb69bcdf4697f683f2ecf15b9865d8`，full_state_v2 serving 基线）。
- `TRAIN_MJ_ROOT` 指向 v1.3.0 的只读 checkout；train_mj 发布新 tag 时重跑 A0-4 契约探测与 A2 后再决定是否升级（开发计划 §7）。

## 7. 多后端基线

上文保留 v1.3.0 的已联调基线；它不是所有后端共享的全局契约。新增 v1.2.0 的代码身份、维度、旧版字段语义及部署校准见 [多后端配置契约](multi-backend-contract.md)，两个版本的权重哈希见 configs/arena_selfplay.yaml。

A0 验证旧版源码与权重文件身份；A1 已完成双服务单局及 profile smoke，见 [A1 记录](milestones/A1.md)。A2 普通 XML 双路径通过，冻结 tag 保留 DEF-001 特定场景失败，见 [A2 记录](milestones/A2.md)。按用户确认，该已知例外不阻塞 A3 工程验收；A3 保存批量实例清单。默认 pass_bias=null 时，旧/新服务端默认值各自生效，不能共享一份全局校准。
