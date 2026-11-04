# Windows：从干净环境到本地对弈、线上排位

本指南面向 **Windows x64 + CPython 3.11.x + PowerShell**，仅需预装 Python，之后用标准库 `venv` 安装依赖。默认 CPU 推理，无需 Anaconda、uv、CUDA 或训练环境。所有命令从 `riichi_lab` 仓库根目录执行；每条命令成功后再执行下一条。**Linux 用户请使用 [Linux 使用指南](getting-started-linux.md)**。自动启动脚本统一使用调用它的 Python 解释器启动服务及适配层。

流程：安装适配层 → 准备固定版本的 train_mj 和权重 → 安装 CPU 推理依赖 → 本地单局 → 平台验证 → 单局有界排位。这里的本地对弈是模型控制四席的自动对局，当前 CLI 没有人类交互打牌界面。

## 1. 准备 Python 和仓库

```powershell
python --version
python -c "import struct, sys; print(sys.executable); print(struct.calcsize('P') * 8); assert sys.version_info[:2] == (3, 11)"
```

应为 Python 3.11.x、64 位。本项目 `pyproject.toml` 限定 `>=3.11,<3.12`，不能仅因平台官网示例使用 Python 3.12+ 就升级。版本不符时先安装 [Python 3.11 的 Windows x64 发行版](https://www.python.org/downloads/windows/)，再用其完整 `python.exe` 路径创建 venv。后续都通过 venv 内的解释器运行。

还需要 [Git for Windows](https://git-scm.com/downloads/win)：本项目的版本核验、源码归档和运行清单会调用 Git，单独下载源码 ZIP 不足以运行这些入口。安装后重新打开 PowerShell，检查 `git --version`。

选择一个自己的工作目录（无需复用开发机盘符），将两个仓库放在同一级：

```powershell
git clone https://github.com/RLWS/riichi_lab.git
git clone https://github.com/RLWS/train_mj.git
Set-Location riichi_lab
git -C ../train_mj fetch --tags
```

已有仓库时跳过 clone。仓库访问权限由所属账号提供。模型权重是额外文件，Git clone 不会自动带来下面的 `runs/` 权重。

## 2. 用 venv 安装适配层

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m rilab.cli --help
```

无需执行 `Activate.ps1`，因此也无需修改 PowerShell 执行策略。不要使用 `--system-site-packages`。项目直接依赖已固定版本：riichienv 0.4.11、httpx 0.28.1、websockets 17.4、PyYAML 6.0.3。`uv.lock` 留作既有依赖记录；本指南的 pip 不读取它，也不声称所有传递依赖都已锁定。

可选：先跑不需要权重或平台 token 的回归测试：

```powershell
.\.venv\Scripts\python.exe -m pip install pytest==9.2.0
.\.venv\Scripts\python.exe -m pytest -q
```

## 3. 准备固定源码与真实权重

默认本地单局比较 v1.3.0 与 v1.2.0，两个版本都要准备。校验 tag 指向：

```powershell
git -C ../train_mj rev-parse 'v1.2.0^{commit}'
git -C ../train_mj rev-parse 'v1.3.0^{commit}'
```

依次应为 `758981ab9fb457c882ab9c577c9377976f97f42b`、`1e5092f344fb69bcdf4697f683f2ecf15b9865d8`。无需切换或修改 train_mj 当前工作区：后面的自动入口会从核对过的 tag 导出隔离源码。

将可信来源的四份权重复制到 **train_mj 根目录下**，保留以下相对路径；本仓库不提供权重下载链接，也不自动训练或生成替代权重：

| 版本 / 用途 | train_mj 内相对路径 | SHA256 |
|---|---|---|
| v1.2.0 turn | `runs/2026-06-24_p800_multitask_v1_1_ddp_40w/multitask/checkpoints/best_turn.pt` | `532ca85655e42b37d037688e8ca71382b3f425e6f9057c42429b471c4d1c8dd7` |
| v1.2.0 response | `runs/2026-06-24_p800_multitask_v1_1_ddp_40w/multitask/checkpoints/best_response.pt` | `3c9b43d3ac2ff0ac5a9b1e90a2a9a32bf1057f4d7f823f90bdc0caa3ce39e2d4` |
| v1.3.0 turn | `runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt` | `30f2811cd86beebad57d4da1428272b25b598fb64a6231b5a1760e11ec08bf90` |
| v1.3.0 response | `runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_response.pt` | `e1d8c58a2aa61cef23852f15fe5bb9c65014bc3916c52756fb8c4cd1cd956c51` |

手工核对示例：

```powershell
Get-FileHash ../train_mj/runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt -Algorithm SHA256
```

没有对应权重时可以完成安装、回归测试和排位 dry-run，不能完成真实模型对弈。换权重或 tag 应重新做兼容性验收，不能只改配置中的哈希冒充既有部署。

## 4. 安装 CPU 推理依赖

适配层通过 HTTP 调模型；`pip install -e .` 本身不安装 PyTorch 或模型服务。在同一个 venv 补齐 CPU 服务依赖即可：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements/inference.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, numpy, fastapi, uvicorn; print(torch.__version__); print(torch.rand(1))"
```

CPU wheel 来源为 [PyTorch 官方安装源](https://pytorch.org/get-started/locally/)。CPU 包下载较大，等待安装完成；本指南固定 2.5.1 以贴近原权重运行基线。`requirements/inference.txt` 固定额外推理依赖，无需安装 train_mj 的训练、数据处理和 GPU 工具，也无需 `pip install -e ../train_mj`：服务脚本会把导出的版本源码加入 import 路径，避免两个版本混用。

## 5. 先完成本地单局

```powershell
$localRun = "outputs/local_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
.\.venv\Scripts\python.exe scripts/run_a1_integration.py --train-mj-root ../train_mj --output-dir $localRun
if ($LASTEXITCODE -ne 0) { throw "本地对局失败，检查 $localRun" }
Get-Content "$localRun/game/summary.json"
```

脚本核对两个 tag 和四份权重哈希，导出源码，启动两个 CPU HTTP 服务并自动选择空闲端口，运行 seed=1 的一个半庄（v1.3.0 对三个 v1.2.0），最后回收服务。无需手工开端口或运行服务。应看到 `valid=true`、四席 `scores` 和 `ranks`；`verification.json` 包含该单局 summary。出错看 `v1_2_0.log`、`v1_3_0.log` 和 `game/summary.json`。

输出目录必须是新目录，不覆盖、不断点续跑。单局子进程上限 300 秒，低性能 CPU 可能触发超时，应先检查日志和机器负载。已有完整日志不能只凭进程退出码替代审查。

需要完整工程日程时：

```powershell
$batchRun = "outputs/batch_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
.\.venv\Scripts\python.exe scripts/run_a1_integration.py --train-mj-root ../train_mj --output-dir $batchRun --batch --batch-runs 1
```

默认配置为 8 种子、112 半庄；`--batch-runs 2` 会再跑一遍核对重放一致性，成本翻倍。它是工程检查。正式强度实验另见 [强度运行说明](run-strength.md)，不要把旧实验种子重跑当新增独立证据。

## 6. 平台验证：新机器人先激活

平台要求先用 GitHub 登录 [riichi.dev](https://riichi.dev/)，在 [Bots 页面](https://riichi.dev/bots) 创建机器人并保存只显示一次的 token。新机器人经东风验证后激活，之后才能排位，见 [官方验证流程](https://riichi.dev/docs/validation) 和 [官方排位说明](https://riichi.dev/docs/ranked)（2026-11-04 核对）。已经 active 的机器人无需再次调用仅供 pending 状态的验证端点。

输入 token 到当前 PowerShell 进程环境，不把 token 写进配置或命令历史：

```powershell
$env:RIICHI_DEV_BOT_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Bot token' -AsSecureString)).Password
$validationRun = "outputs/validate_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
.\.venv\Scripts\python.exe scripts/run_online_validation.py --train-mj-root ../train_mj --output-dir $validationRun
if ($LASTEXITCODE -ne 0) { throw "平台验证失败，检查 $validationRun" }
Get-Content "$validationRun/review.json"
```

此入口固定 v1.3.0，只需该版本两份权重。它校验 tag/权重，自动启动隔离的回环 CPU 服务，去除该临时配置的 HTTP key 要求，并执行一次平台验证与离线审查，结束后回收服务。平台 token 不传给模型服务进程。

检查 `session/summary.json` 的 `completed=true`、`faults=[]`、`validation_result.passed=true`，以及 `review.json` 的 `b1_protocol_passed=true`、ack、fallback 和耗时。平台通过仍可能发生默认动作；首次排位前应确认零 fallback/defaulted 并审查延迟。验证脚本最多等待 900 秒，不循环重试。

## 7. 线上排位：独立启动服务，再跑一个有界批次

排位 CLI 不会自动启动模型服务。首次使用下面的 v1.3.0 / 8000 端口流程，与默认 `arena_ranked.yaml` 对齐；不要直接套用 `arena_ranked_v120.yaml` / `arena_ranked_v130.yaml`，它们使用 8012/8013、独立 token/key 变量，且局数为 100 的占位计划。

在**终端 A**（仓库根目录）导出固定源码并再次核验权重，然后启动服务、保持终端运行。这样已 active、跳过第 6 节的机器人也可以独立部署：

```powershell
$deployDir = "outputs/deploy_v130_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory $deployDir | Out-Null
if ((git -C ../train_mj rev-parse 'v1.3.0^{commit}') -ne '1e5092f344fb69bcdf4697f683f2ecf15b9865d8') { throw 'v1.3.0 tag 不符' }
git -C ../train_mj archive --format=zip --output "$((Resolve-Path $deployDir).Path)/source.zip" v1.3.0
if ($LASTEXITCODE -ne 0) { throw '源码导出失败' }
Expand-Archive "$deployDir/source.zip" "$deployDir/source"
.\.venv\Scripts\python.exe -c "import hashlib, pathlib, yaml; src=pathlib.Path(r'$deployDir/source'); c=yaml.safe_load((src/'configs/api_local.yaml').read_text(encoding='utf-8')); ids=yaml.safe_load(pathlib.Path('configs/arena_ranked.yaml').read_text(encoding='utf-8'))['backend']['model_identity']; assert all(hashlib.sha256((pathlib.Path('../train_mj')/c['models'][t]['checkpoint']).read_bytes()).hexdigest()==ids[t+'_sha256'] for t in ('turn','response')), 'weight hash mismatch'; print('Weights verified')"
if ($LASTEXITCODE -ne 0) { throw '权重核验失败' }
.\.venv\Scripts\python.exe scripts/serve_integration_backend.py --source-root "$deployDir/source" --weights-root ../train_mj --port 8000
```

该辅助服务固定 CPU、单线程、`127.0.0.1`、无 HTTP 鉴权，保留 tag 内的模型校准。它从 train_mj 读取原权重，核验后不要替换这些文件。它适用于本机流程，不是公开网络部署入口。

在**终端 B**进入同一仓库，重新安全输入同一 active 机器人的 token（环境变量不跨终端共享），准备批次配置：

```powershell
$env:RIICHI_DEV_BOT_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Bot token' -AsSecureString)).Password
New-Item -ItemType Directory -Force outputs | Out-Null
.\.venv\Scripts\python.exe -c "import pathlib, yaml; c=yaml.safe_load(pathlib.Path('configs/arena_ranked.yaml').read_text(encoding='utf-8')); c['backend'].pop('api_key_env', None); pathlib.Path('outputs/ranked.local.yaml').write_text(yaml.safe_dump(c), encoding='utf-8')"
.\.venv\Scripts\python.exe -m rilab.cli probe --api-base http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m rilab.cli arena-online --mode ranked --config outputs/ranked.local.yaml --dry-run
```

本机无鉴权服务对应的本地配置需**删除** `backend.api_key_env`，否则客户端会因缺少 `TRAIN_MJ_API_KEY` 报错。平台 token 与 HTTP key 是两种凭证，不可互用。dry-run 不连接模型或平台，也不验证 token、权重和线上兼容性；probe 只检查存活与元信息结构，不验证实际权重哈希。

打开 `outputs/ranked.local.yaml` 检查计划。默认是最多 **1 局 / 总计 7200 秒 / 单会话 7200 秒 / 匹配 300 秒**，零 fallback/defaulted 配额；延迟上限 2500ms、最小时间库 1000ms、最小期限余量 500ms。确认这些预算后显式开始真实排位：

```powershell
$rankedRun = "outputs/ranked_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
.\.venv\Scripts\python.exe -m rilab.cli arena-online --mode ranked --config outputs/ranked.local.yaml --output-dir $rankedRun
$rankedExit = $LASTEXITCODE
Get-Content "$rankedRun/summary.json"
Write-Output "Exit code: $rankedExit"
```

只有完整完成预算且未触发故障或停止门槛才退出 0（`status=completed_budget`）。`stopped`、`failed`、`cancelled` 均需检查原因，不能把已完成某一局等同于批次成功。保存整个目录：`manifest.json`、`summary.json`、`game_0001/session.jsonl`、逐局 `summary.json` 和 `review.json`。

门槛一般在当前局结束后阻止新局；匹配/会话/总时长是硬上限。排位断线后平台可能继续以默认动作完成并计入评级，见 [官方断线说明](https://riichi.dev/docs/ranked)，因此不要把 Ctrl+C 当成无影响暂停。当前工具不支持断线恢复或旧目录续跑。完成后可在终端 A 用 Ctrl+C 关闭服务，在终端 B 清除 token：

```powershell
Remove-Item Env:RIICHI_DEV_BOT_TOKEN
```

扩大局数前先审查首批日志，再修改本地配置的 `ranked` 预算，使用新输出目录。不要在批次运行中替换权重、改校准或追改门槛。已知 B1 网络存在长尾，不能保证任意网络下无默认动作；详细停止语义见 [B2 记录](milestones/B2.md)。

## 8. 常见问题与验收边界

| 现象 | 处理 |
|---|---|
| 找不到 Python / 版本错误 | 使用 CPython 3.11 x64 的完整解释器路径重新创建新 venv；不要复用从别处复制的 venv。 |
| 没有 pip | 用该解释器执行 `python -m ensurepip --upgrade` 后重试。 |
| pip 找不到固定版本或下载失败 | 核对解释器、网络、镜像同步；必要时为普通包显式加 `--index-url https://pypi.org/simple`。PyTorch 保持官方 CPU 源，不随意放宽版本。 |
| 缺少 Git / tag | 安装 Git，重新打开终端并 fetch tags；核对上文 commit。 |
| 缺少 `.pt` / 哈希不符 | 按表恢复原权重和目录，勿改声明哈希绕过校验。 |
| `ModuleNotFoundError: torch/fastapi` | 第 4 节必须在运行服务的同一个 venv 执行。 |
| 端口占用 / 连接被拒绝 | 确认终端 A 服务活着；8000 如已占用，换空闲端口并同步改本地 YAML 的 `backend.api_base`。 |
| 缺少 key/token 环境变量 | 检查 YAML 的变量名及当前终端；无鉴权回环服务删除 `api_key_env`，不要填平台 token。 |
| inactive / pending / token 错误 | 按 Bots 页面状态走对应流程；pending 先验证，active 直接排位；勿并发使用同一 bot。 |
| stopped / fallback / defaulted | 保留完整日志，检查网络、服务器 ack、模型延迟和停止原因；先排除问题再开新批次。 |
| 输出目录已存在 | 使用新目录；不删掉失败记录继续凑局数。 |

本指南沿用已登记的 v1.2.0/v1.3.0 和权重。冻结版本仍有 DEF-001 已知训练采样缺陷，修复分支不会自动修复既有权重，见 [兼容性记录](compatibility.md)。既有真实平台验证和本地模拟通过，不等于这台新机器已经完成排位验收；需要以上步骤的实际产物证明。

环境跑通后可用 `python -m pip freeze`（使用 venv 内解释器）保存部署依赖清单，并保留 Git commit、配置与运行目录。不要提交 venv、token 或模型权重。

## 9. 本次指南验证记录

2026-11-04，Windows x64 / Python 3.11.5：在 `outputs/setup_check/.venv` 新建 venv（未启用系统 site-packages），通过 pip 从头安装适配层、pytest、CPU PyTorch 2.5.1 和本指南推理依赖。`pip check` 通过；178 项测试通过，零 skipped；排位与强度计划 dry-run 通过。使用该新 venv 同时运行两个隔离 HTTP 服务与适配层，核对固定 tag/四份权重后完成 seed=1 的真实模型半庄，`valid=true`，629 步、634 次决策。

这是现有 Windows 主机上隔离 venv 的验证，未重装操作系统；Git 和模型文件按上文作为外部输入准备。本次没有使用 token 连接平台验证或真实排位，线上说明依据现有 B1/B2 实现、验收记录和官方文档。
