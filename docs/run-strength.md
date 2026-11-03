# 手动启动强度对局

2026-11-03。**本页的预设实验已由用户完成，正式统计见 [A4 正式强度验收](milestones/A4-formal.md)。** 下文保留启动方式；相同种子的重跑仅用于复现，不能算新的独立样本。后续独立实验需另定未使用的种子与协议。

## 预设实验

- v1.3.0 候选对比 v1.2.0 基线，固定 tag/commit 和既有四份权重。各用原服务校准，比较完整部署策略；DEF-001 保留为已知例外。
- [arena_strength.yaml](../configs/arena_strength.yaml) 使用独立种子 10001–10128，排除 A3 工程种子 1–8。每种子 4 候选焦点局、4 控制焦点局、6 个 2v2 局，共 128 个种子簇、1,792 个半庄；只跑一轮。
- [strength_protocol.yaml](../configs/strength_protocol.yaml) 固定预算与停止规则。平均顺位候选减基线的 95% 簇 bootstrap 区间上界须 ≤ −0.05，且末位率差区间上界须 ≤ +0.02，才认定有意义提升；顺位差区间下界 ≥ +0.05 则认定有意义回退，其余为证据不足。2v2 独立报告，不混入主实验。
- 同机 A3 的 112 半庄耗时约 10–11 分钟，估计本轮约 3 小时，实际取决于负载。整体超时默认 24 小时。固定首轮预算不保证达到统计判别力。

可以在启动前调整预算或门槛，但配置与协议中的种子/簇数/半庄数必须一致；启动后不能根据中途成绩改模型、种子、门槛或预算。

## PowerShell 命令

在当前这台机器执行：

```powershell
Set-Location 'F:\Code\AI\train_study\riichi_lab'
$runDir = "outputs/strength_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& 'D:\Anaconda3\python.exe' scripts/run_strength.py --output-dir $runDir
```

这里使用已安装 torch/FastAPI/uvicorn 的训练开发 Python。脚本自动调用项目 .venv 运行适配层，不需要手动开服务或设置端口。保持电脑运行、终端开启。

只检查计划、不启动服务或对局：

```powershell
& 'D:\Anaconda3\python.exe' scripts/run_strength.py --output-dir outputs/check_only --dry-run
```

dry-run 只核对计划结构，不创建目录；实际启动还核对 tag、四份权重 SHA256 和服务实际校准。默认从相邻 train_mj 读取代码与权重，可用 --train-mj-root 指定。输出目录必须不存在，时间戳命名避免覆盖旧结果。

## 登记、进度与完成

开局前生成 protocol.registered.json，保存协议、实验配置、权重部署核验结果、实际服务元信息、登记时间和配置/源码哈希。登记文件的哈希进入 batch manifest。校准不符、已有 batch 目录或试图覆盖登记时拒绝启动，不在观察结果后补登记。

终端逐局打印 completed/total。运行期间 summary.valid=false 表示尚未完成；不据此把进行中的运行误判为失败，也不提前拿局部结果下结论。

完成后保留整个输出目录，通知我目录路径即可。主要产物：

- verification.json：应为 valid=true、runs=1；replay_identical=null 表示本次没有重复运行。
- protocol.registered.json：开局前封存的协议及部署信息。
- batch/manifest.json、batch/summary.json：日程与结果；summary 应为 valid=true、completed_games=1792。
- batch/games/、batch/decisions/、batch/results/：事件、压缩决策分片和逐局结果。
- v1_2_0.log、v1_3_0.log：两个隔离服务日志。

推理、状态、非法动作、写盘等错误会停止开启新局并保留产物。失败或中断时也通知我原目录，不删掉错误局继续凑数量。当前入口不提供断点续跑。正常或异常退出都会回收脚本启动的服务。

arena-evaluate 默认生成探索性报告；加入 --formal 会核验封存协议、预算、部署身份和完整产物，离线重建所有输入与动作，再使用封存参数进行正式统计。原始运行不被修改，也不会调用模型或启动训练。命令见 A4 正式验收记录。
