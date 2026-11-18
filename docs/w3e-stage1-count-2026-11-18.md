# W3e Stage 1 count-phase report (2026-11-18)

- Status: **count-only phase complete**; outcome analysis was run later, see `train_mj/docs/w3e-turn-riichi-result-2026-11-18.md`. This document remains the count-only snapshot.
- Branch: `codex/w3e-turn-riichi`
- Batch code commit: `283fb47852d2447027685d6d31bb4a61e48a8ae3`
- Plan commit: `37861ad` plus doc-only clarification `19bcc7d`
- Plan SHA-256: `ea7db4c426f991f7d94a420e6463c8b25597f897c4e72c7084ceb6e7def65379`
- Batch directory: `outputs/w3e_turn_20261118/stage1/batch/`
- Count directory: `outputs/w3e_turn_20261118/stage1/`

This report contains **event counts only**. No win, deal, income, loss, or hand-outcome
field was read or computed at this phase. The extension decision is therefore based
only on the pre-registered `B_extra` unique-hand count.

---

## 1. Batch integrity

| Item | Value |
|---|---|
| seeds | `43001-43064` (64 seed clusters) |
| arms | A = v130/v130; B = v140/v130 (old Response fixed) |
| rotations per seed/arm | 4 |
| expected games | 512 |
| completed games | 512 |
| fallbacks | 0 |
| audit games | 512 |
| `verification.json valid` | `true` |
| `rilab_commit` | `283fb47852d2447027685d6d31bb4a61e48a8ae3` |
| `rilab_dirty` | `false` |
| `replay_digest` | `f3ec389af9d3c2bebff96b51ecd0d1fda95a74fc372454d17560cbb3d795d7fb` |
| protocol registration SHA-256 | `9be789b93b90471c28d04044d8c628eb300f5e30cdfa2d70e0e55c7b5ce378b3` |
| smoke seeds | `43501-43502` (engineering only, excluded) |
| forbidden seeds used | none (`40001-40008`, `41001-41128`, `42001-42128` avoided) |

---

## 2. Captured riichi declarations

Only focal-seat `turn` decisions whose executed action is `riichi_discard` were
captured with full state and replayed against the other Turn service.

| Item | A (old Turn) | B (new Turn) | Total |
|---|---:|---:|---:|
| captured executed riichi hands | 790 | 682 | 1,472 |
| both versions would riichi | 709 | 648 | 1,357 |
| executed riichi where the other Turn would discard | **81** (`A_mirror`) | **34** (`B_extra`) | 115 |
| share of that arm's captured riichi hands | 10.25% | 4.99% | 7.81% |

Replay integrity:

| Item | Value |
|---|---:|
| captured records | 1,472 |
| captured with full state | 1,472 |
| classified by both services | 1,472 |
| excluded by common candidate support | 0 |
| missing state | 0 |
| missing candidates | 0 |
| shadow replay errors | 0 |

`A_mirror` = old Turn declared riichi where the new Turn would discard.
`B_extra` = new Turn declared riichi where the old Turn would discard.

---

## 3. Risk-concentration counts (no outcomes)

| Subset | Unique hands |
|---|---:|
| `B_extra` with `opp_riichi_late` | 14 |
| `A_mirror` with `opp_riichi_late` | 12 |

These are counts only; no deal/win/income/loss rate has been computed.

---

## 4. Pre-registered extension decision

The extension rule in `configs/w3e/w3e_turn_riichi_plan.yaml` is:

> extend to Stage 2 (`43065-43128`) only if unique `B_extra` hands < 24.

Observed: **34 unique `B_extra` hands**, which is `>= 24`.

Decision:

- **Stage 2 is not started.**
- **No extension sample is used.**
- The Stage 1 sample is the pre-registered analysis sample.
- The decision is based only on `counts.json` and does not use any outcome rate.

This follows the registered rule exactly. Seeds `43065-43128` remain unused.

---

## 5. No-outcome checks

- `outputs/w3e_turn_20261118/stage1/counts.json` has `outcomes_inspected: false`.
- `outputs/w3e_turn_20261118/stage1/events.jsonl` contains 1,472 rows and no
  `win`, `deal`, `income`, `loss`, `hand`, `win_rate`, or `deal_rate` fields.
- `scripts/run_w3e_turn_riichi.py --phase analyze` was **not run at the time this count snapshot was written**; it was run later after the extension decision, see the result document.
- The W2 formal verdict, v1.3.0 default, and v1.4.0 candidate status are unchanged.

---

## 6. Artifacts and commands

```powershell
cd F:\Code\AI\train_study\riichi_lab

# Stage 1 batch (completed)
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage1 --phase run --output-root outputs\w3e_turn_20261118

# Count-only shadow classification (completed; this report)
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage1 --phase count --output-root outputs\w3e_turn_20261118

# Outcome analysis is intentionally NOT run in this phase.
```

Artifacts:

```text
outputs/w3e_turn_20261118/stage1/
  protocol.registered.json
  native_config.json
  batch/verification.json
  batch/summary.json
  batch/manifest.json
  batch/games/*.mjai.jsonl
  batch/decisions/*.jsonl.gz
  batch/results/*.json
  events.jsonl        # 1,472 captured states, no outcomes
  counts.json         # event counts only, no outcomes
```

---

## 7. Next step

Run the pre-registered Stage 1 outcome analysis:

```powershell
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage1 --phase analyze --output-root outputs\w3e_turn_20261118
```

No plan, seed, arm, endpoint, or threshold change is permitted before that analysis.
