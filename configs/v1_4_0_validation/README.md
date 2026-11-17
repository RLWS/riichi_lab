# W2 Turn x Response 2x2

`w2_plan.yaml` is the pre-registration input for the W2 component experiment.

- A: old Turn + old Response (all-v1.3.0 control)
- B: new Turn + old Response
- C: old Turn + new Response
- D: new Turn + new Response (complete v1.4.0)
- D-A is the only formal comparison.
- B-A, C-A, D-B, D-C and the derived component/interaction contrasts are exploratory and unadjusted.
- Smoke seeds: `40001, 40002`; formal seeds: `42001-42128`; the runner asserts that the two sets are disjoint.

The two service configs use checkpoint paths relative to `../train_mj` (`weights_root` in the plan). The launcher starts isolated services on free local ports; `api.port` in the YAML is overridden.

## Python environment

The service subprocess imports `torch` and `riichienv`. The `riichi_lab/.venv` used for some client-only tests does not contain `torch`; run the validation entry point with an interpreter that has both, for example:

```powershell
D:\Anaconda3\python.exe scripts\run_v140_validation.py ...
```

The runner passes `sys.executable` to the service subprocess, so the same interpreter is used for the client and both services.

## Commands

```powershell
cd F:\Code\AI\train_study\riichi_lab

# 1) Validate the plan and schedule; no services or games.
D:\Anaconda3\python.exe scripts\run_v140_validation.py `
  --plan configs\v1_4_0_validation\w2_plan.yaml `
  --stage 2x2 --smoke --output-dir outputs\w2_v140_smoke --dry-run

# 2) Smoke: 2 seeds x 4 arms x 4 rotations = 32 half-games.
D:\Anaconda3\python.exe scripts\run_v140_validation.py `
  --plan configs\v1_4_0_validation\w2_plan.yaml `
  --stage 2x2 --smoke --output-dir outputs\w2_v140_smoke

# 3) Formal: 128 seeds x 4 arms x 4 rotations = 2,048 half-games.
D:\Anaconda3\python.exe scripts\run_v140_validation.py `
  --plan configs\v1_4_0_validation\w2_plan.yaml `
  --stage 2x2 --formal --output-dir outputs\w2_v140_20261118 `
  --smoke-dir outputs\w2_v140_smoke\smoke
```

The formal stage refuses to start unless `smoke/verification.json` is valid and has the expected 32 games. The formal output contains:

```text
outputs/w2_v140_20261118/
  protocol.registered.json
  native_config.json
  batch/manifest.json
  batch/summary.json
  batch/verification.json
  batch/games/
  batch/decisions/
  batch/results/
  report.json
  report.md
```

The runner does not support changing seeds, arms, calibration or thresholds after starting. A failure leaves the stage directory in place; do not overwrite it. Resume is limited to infrastructure interruption with unchanged model/service/config/rule/schedule identity and must use a separate `resume_manifest.json` workflow.
