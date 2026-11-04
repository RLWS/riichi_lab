# Linux：从干净环境到本地对弈、线上排位

面向 **Linux x86_64 / glibc / CPython 3.11.x / Bash**，以仅预装 Python 的环境为起点，使用标准库 `venv` 和 CPU 推理。无需 Anaconda、uv、CUDA 或训练环境。Windows 用户见 [PowerShell 指南](getting-started.md)。以下每条命令成功后再继续；安装完成后，所有运行命令从 `riichi_lab` 仓库根目录执行。

本地对弈是模型控制四席的自动对局，当前 CLI 没有人类交互打牌界面。Alpine/musl、ARM 和其他架构不在本指南验证范围；固定依赖若无对应 wheel，不要盲目升级或转源码编译来跳过兼容性检查。

## 1. 检查系统、Python 和 Git

```bash
uname -m
python3.11 --version
python3.11 -c 'import struct, sys; print(sys.executable); print(struct.calcsize("P") * 8); assert sys.version_info[:2] == (3, 11)'
git --version
```

应为 `x86_64`、Python 3.11.x、64 位。本项目限制 `>=3.11,<3.12`；`python3` 的默认版本未必符合要求。如果已有符合条件的 Python 使用其他名称，可将创建 venv 时的 `python3.11` 换成该解释器的绝对路径。

还需 Git、系统 CA 证书及该 Python 的 venv/ensurepip 支持。Debian/Ubuntu 可先安装通用系统工具：

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates
```

若 `python3.11 -m venv` 提示缺少 ensurepip，在软件源提供匹配版本时执行 `sudo apt-get install python3.11-venv`；不要用其他 Python 版本的 venv 包替代。没有 Python 3.11 的发行版需先通过可信发行渠道准备 3.11，再继续。不要替换系统 `/usr/bin/python3`，也无需使用 `sudo pip`。venv 会在 Linux 创建 `bin/`，与 Windows 的 `Scripts/` 不同，见 [Python 官方说明](https://docs.python.org/3.11/library/venv.html)。

在自己的工作目录获取两个同级仓库：

```bash
git clone https://github.com/RLWS/riichi_lab.git
git clone https://github.com/RLWS/train_mj.git
cd riichi_lab
git -C ../train_mj fetch --tags
```

已有仓库则跳过 clone。仓库访问权限需自行具备。若从 Windows 迁移，只迁移代码和权重，在 Linux 重新创建 venv；不能复制 Windows 的 `.venv` 使用。

## 2. 创建 venv、安装适配层及 CPU 推理依赖

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements/inference.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch, numpy, fastapi, uvicorn; print(torch.__version__); print(torch.rand(1))'
.venv/bin/python -m rilab.cli --help
```

无需 `source .venv/bin/activate`；显式指定解释器可以避免混用系统环境。不要加 `--system-site-packages`。适配层不自带 PyTorch，CPU wheel 从 [PyTorch 官方源](https://pytorch.org/get-started/locally/) 安装，下载较大。无需安装 train_mj 的训练/数据处理依赖，也无需 `pip install -e ../train_mj`，服务会导入对应 tag 的隔离源码。

自动启动脚本使用 `sys.executable` 调用服务、适配层及离线审查，因此它们必须通过同一个已经安装全部依赖的 venv 启动。强度入口也遵循此规则，不依赖名为 `.venv` 的固定路径。

适配层直接依赖与额外推理依赖固定版本；pip 不读取 `uv.lock`，传递依赖没有全部锁定。跑通后可执行 `.venv/bin/python -m pip freeze > outputs/environment.txt` 保存实际部署清单（先创建 outputs 目录）。

可选回归测试，不需要权重或平台 token：

```bash
.venv/bin/python -m pip install pytest==9.2.0
.venv/bin/python -m pytest -q
```

## 3. 准备固定 tag 与模型权重

```bash
git -C ../train_mj rev-parse 'v1.2.0^{commit}'
git -C ../train_mj rev-parse 'v1.3.0^{commit}'
```

依次应为 `758981ab9fb457c882ab9c577c9377976f97f42b`、`1e5092f344fb69bcdf4697f683f2ecf15b9865d8`。自动入口从核对过的 tag 导出源码，无需 checkout 或修改 train_mj 当前工作区。

Git clone 不会带来实际权重。将可信来源的四份文件放在 **train_mj 根目录下**，保持以下路径；本仓库不提供权重下载，也不自动训练替代权重：

| 版本 / 用途 | train_mj 内相对路径 | SHA256 |
|---|---|---|
| v1.2.0 turn | `runs/2026-06-24_p800_multitask_v1_1_ddp_40w/multitask/checkpoints/best_turn.pt` | `532ca85655e42b37d037688e8ca71382b3f425e6f9057c42429b471c4d1c8dd7` |
| v1.2.0 response | `runs/2026-06-24_p800_multitask_v1_1_ddp_40w/multitask/checkpoints/best_response.pt` | `3c9b43d3ac2ff0ac5a9b1e90a2a9a32bf1057f4d7f823f90bdc0caa3ce39e2d4` |
| v1.3.0 turn | `runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt` | `30f2811cd86beebad57d4da1428272b25b598fb64a6231b5a1760e11ec08bf90` |
| v1.3.0 response | `runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_response.pt` | `e1d8c58a2aa61cef23852f15fe5bb9c65014bc3916c52756fb8c4cd1cd956c51` |

手工检查示例：

```bash
sha256sum ../train_mj/runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt
```

默认双版本本地对局需要全部四份；只使用 v1.3.0 做平台验证/排位则需要其两份。没有权重仍可安装、跑测试和 dry-run，不能完成真实模型对局。Linux 路径区分大小写；不要改声明哈希绕过缺文件或版本不符。

## 4. 本地单局与批量对弈

```bash
local_run="outputs/local_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python scripts/run_a1_integration.py --train-mj-root ../train_mj --output-dir "$local_run"
cat "$local_run/game/summary.json"
```

自动核对两个 tag 与四份权重，导出源码，启动两个回环 CPU HTTP 服务并选择空闲端口，完成 seed=1 的半庄（一个 v1.3.0 对三个 v1.2.0），退出时回收服务。应看到 `valid=true` 以及四席 `scores`、`ranks`。单局子进程上限 300 秒，低性能 CPU 需留意超时。失败时保留整个目录，检查 `v1_2_0.log`、`v1_3_0.log`、`game/summary.json`，不要仅凭产生了牌谱判断成功。

需要工程批量检查时：

```bash
batch_run="outputs/batch_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python scripts/run_a1_integration.py --train-mj-root ../train_mj --output-dir "$batch_run" --batch --batch-runs 1
```

默认 8 种子、112 半庄；`--batch-runs 2` 再跑一遍检查一致性，成本翻倍。所有输出目录必须不存在；不覆盖、不续跑。正式强度流程见 [强度运行说明](run-strength.md) 的 Bash 命令；旧种子重跑不算新增独立证据。

## 5. 新机器人先做平台验证

在 [riichi.dev](https://riichi.dev/) 用 GitHub 登录，在 [Bots 页面](https://riichi.dev/bots) 创建机器人并保存 token。pending 机器人先完成东风验证才会 active；已 active 的机器人跳过验证，直接按下一节部署排位。依据 [官方验证说明](https://riichi.dev/docs/validation) 与 [排位说明](https://riichi.dev/docs/ranked)（2026-11-04 核对）。

在 Bash 中隐藏输入，避免将 token 放进命令历史或配置：

```bash
read -r -s -p 'Bot token: ' RIICHI_DEV_BOT_TOKEN
printf '\n'
export RIICHI_DEV_BOT_TOKEN
validation_run="outputs/validate_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python scripts/run_online_validation.py --train-mj-root ../train_mj --output-dir "$validation_run"
cat "$validation_run/review.json"
```

此入口固定 v1.3.0，自动校验 tag/两份权重、启动隔离的无 HTTP 鉴权回环服务、执行一次验证及离线审查，并回收服务。平台 token 不传给模型服务进程。最长等待 900 秒，不自动重试。

检查 `session/summary.json` 的 `completed=true`、`faults=[]`、`validation_result.passed=true`，以及 `review.json` 的 `b1_protocol_passed=true`、ack、fallback 和耗时。平台激活不代表零超时，首次排位前仍要检查零 fallback/defaulted 和延迟。

## 6. 有界排位：终端 A 启动服务

排位 CLI 不自动启动模型服务。以下使用 v1.3.0、端口 8000，与 `arena_ranked.yaml` 对齐。`arena_ranked_v120.yaml` / `arena_ranked_v130.yaml` 使用 8012/8013、独立 token/key 变量和 100 局占位预算，不能直接套用。

在终端 A 进入仓库根目录，导出源码、核验权重，启动服务：

```bash
deploy_dir="outputs/deploy_v130_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs
mkdir "$deploy_dir"
test "$(git -C ../train_mj rev-parse 'v1.3.0^{commit}')" = '1e5092f344fb69bcdf4697f683f2ecf15b9865d8'
git -C ../train_mj archive --format=zip --output "$(pwd)/$deploy_dir/source.zip" v1.3.0
.venv/bin/python -m zipfile -e "$deploy_dir/source.zip" "$deploy_dir/source"
.venv/bin/python - "$deploy_dir/source" <<'PY'
import hashlib
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
config = yaml.safe_load((source / 'configs/api_local.yaml').read_text(encoding='utf-8'))
identity = yaml.safe_load(Path('configs/arena_ranked.yaml').read_text(encoding='utf-8'))['backend']['model_identity']
for task in ('turn', 'response'):
    weight = Path('../train_mj') / config['models'][task]['checkpoint']
    if hashlib.sha256(weight.read_bytes()).hexdigest() != identity[f'{task}_sha256']:
        raise SystemExit(f'weight hash mismatch: {task}')
print('Weights verified')
PY
.venv/bin/python scripts/serve_integration_backend.py --source-root "$deploy_dir/source" --weights-root ../train_mj --port 8000
```

务必逐条检查成功，tag 比较返回非零或权重核验报错时停止，不要继续启动。也可将整个部署代码块放入 Bash 脚本并在开头加 `set -euo pipefail`，让出错自动停止。这里使用 Python 标准库解压，无需额外安装 unzip。

服务固定 CPU、单线程、`127.0.0.1`、无 HTTP 鉴权，保留该 tag 的校准；适用于本机双终端流程。不要开放成公网入口。启动后不要替换源码或权重，保持终端 A 运行。

## 7. 终端 B 检查并开始排位

进入同一仓库根目录，在终端 B 输入同一 active 机器人的 token（不同终端不共享刚设置的环境变量），生成本地配置：

```bash
read -r -s -p 'Bot token: ' RIICHI_DEV_BOT_TOKEN
printf '\n'
export RIICHI_DEV_BOT_TOKEN
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path('configs/arena_ranked.yaml').read_text(encoding='utf-8'))
config['backend'].pop('api_key_env', None)
Path('outputs/ranked.local.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
PY
.venv/bin/python -m rilab.cli probe --api-base http://127.0.0.1:8000
.venv/bin/python -m rilab.cli arena-online --mode ranked --config outputs/ranked.local.yaml --dry-run
```

本机服务无 HTTP 鉴权，所以删除本地配置的 `backend.api_key_env`。若接已有鉴权服务，则保留该字段并在当前终端设置其指定的 HTTP key 变量；HTTP key 与平台 token 不能互用。

dry-run 不连接模型或平台，不验证 token/权重；probe 检查存活与元信息结构，不实测权重哈希。打开 `outputs/ranked.local.yaml` 检查预算：默认 **1 局、总计 7200 秒、单会话 7200 秒、匹配 300 秒**；零 fallback/defaulted；延迟上限 2500ms，最小时间库 1000ms，最小期限余量 500ms。确认后执行真实排位：

```bash
ranked_run="outputs/ranked_$(date +%Y%m%d_%H%M%S)"
.venv/bin/python -m rilab.cli arena-online --mode ranked --config outputs/ranked.local.yaml --output-dir "$ranked_run"
ranked_exit=$?
cat "$ranked_run/summary.json"
printf 'Exit code: %s\n' "$ranked_exit"
```

仅完整完成预算且未触发故障或门槛时退出 0，`status=completed_budget`；`stopped`、`failed`、`cancelled` 均需审查原因。保存整个运行目录，含 manifest、批次 summary、`game_0001/session.jsonl`、逐局 summary/review。普通停止门槛在当前局结束后阻止新局，时间上限可关闭当前连接；没有断线重连、续跑或自动补局。

排位期间不要退出终端、断开 SSH、让主机休眠或中途替换权重。若在远程服务器运行，应在已经准备好的持久终端会话（例如 tmux）中完成两终端流程；此指南默认前台运行。断线后平台仍可能用默认动作完成并计入评级，见 [官方说明](https://riichi.dev/docs/ranked)，不能把 Ctrl+C 当暂停。批次结束后在终端 A 用 Ctrl+C 关服务，终端 B 执行 `unset RIICHI_DEV_BOT_TOKEN`。

## 8. 故障检查与边界

| 现象 | 检查与处理 |
|---|---|
| venv / ensurepip 缺失 | 安装与 Python 3.11 匹配的发行版组件；不要 sudo pip 或替换系统 Python。 |
| `externally-managed-environment` | 正在使用系统 pip；改用 `.venv/bin/python -m pip`。 |
| wheel 不匹配 / 找不到固定版本 | 检查 Python 3.11、x86_64、glibc、镜像同步；普通包可显式用 `--index-url https://pypi.org/simple`，PyTorch 用官方 CPU 源。 |
| 找到 Windows `.exe` / 无法执行解释器 | 更新仓库到包含本次 Linux 修复的版本，使用本机新建的 venv；自动入口已改用当前解释器。 |
| 权重不存在 / 哈希不符 | 恢复上表目录和原文件，检查大小写，不修改哈希声明绕过核验。 |
| `torch` / `fastapi` 缺失 | 运行脚本的 venv 必须完成第 2 节全部安装。 |
| 连接被拒绝 / 8000 被占用 | 检查终端 A 服务；换端口时同步修改本地配置 `backend.api_base`。 |
| token / HTTP key 缺失 | 检查当前终端环境和 YAML 变量名，无鉴权回环服务删除 `api_key_env`。 |
| inactive / pending / token 错误 | 按 Bots 页面状态选择验证或排位，勿并发使用同一 bot。 |
| 输出目录已存在 | 换新目录，保留失败产物；不覆盖或续跑。 |
| stopped / fallback / defaulted | 审查网络、ack、模型延迟与停止原因；修复后开新批次。 |

冻结版本仍有 DEF-001 训练采样缺陷，修复分支不会自动修复既有权重，见 [兼容性记录](compatibility.md)。网络长尾与完整停止语义见 [B1](milestones/B1.md)、[B2](milestones/B2.md)。不要提交 token、venv、模型文件或运行产物。

## 9. 本次验证记录

2026-11-04，在 WSL2 Ubuntu 22.04 x86_64 上，用独立 Python 3.11.11 创建未共享系统包的 venv，安装 Linux 版固定依赖与 PyTorch 2.5.1+cpu。因 WSL 下载较慢，Linux wheel 由 Windows 下载后交给 Linux pip 安装；没有使用 Windows 二进制或复制 Windows venv。

对本次文档和启动器改动的隔离源码快照验证：`pip check` 通过；178 项测试全部通过、零 skipped；15 个 Bash 命令块语法通过；排位和强度计划 dry-run 通过。通过 Linux venv 的解释器直接运行 `run_a1_integration.py`，两个真实权重服务与适配层完成 seed=1 半庄，`valid=true`、629 步、634 次决策。该 venv 位于仓库外，确认启动器不再依赖仓库内的 Windows 解释器路径。Windows 上也完成 178 项回归和修改后的自动启动器真实权重半庄。

本次未连接真实平台验证或排位；线上步骤依据既有 B1/B2 实现、验收记录及官方说明。WSL2 的实测不代表所有 Linux 发行版和硬件均已验收。
