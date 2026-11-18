# W5 result: serving-time veto of late extra Riichi under threat

- Status: **pooled analysis complete; decision `underpowered`**
- Date: 2026-11-18
- Branch: `codex/w5-riichi-veto`
- Pre-registration: `docs/w5-riichi-veto-preregistration-2026-11-18.md`
- Plan: `configs/w5/w5_riichi_veto_plan.yaml` (SHA-256 `0e3ba39c29457b690721f189c882c21c350a40db316626aa91dea37f2bee5a38`)
- Batch code commit: `b61509875876ecb6a2ba5394d015a94deae79072`
- Outputs: `outputs/w5_riichi_veto_20261118/`

This was a serving-time intervention, not retraining. No weights, training data,
or calibration changed.

---

## 1. Design recap

| Arm | Turn | Veto | Response |
|---|---|---|---|
| `C` control | v1.4.0 | none | v1.3.0 |
| `T` treatment | v1.4.0 | old-v1.3.0 discard substitution | v1.3.0 |

- Veto condition: `opp_riichi_count > 0`, `turn_number > 12`, v1.4.0 selected
  `riichi_discard`, and v1.3.0 selected `discard` on the same v2-normalized state.
- Three fixed v1.3.0 opponents, four seat rotations per seed and arm.
- Primary endpoint: hand-level deal rate of the first-veto hand, paired by
  `(seed, rotation, hand_id)`.
- Count-only extension: Stage 2 only if Stage 1 first-veto pairs `< 24`.

---

## 2. Batch integrity

| Item | Stage 1 | Stage 2 | Pooled |
|---|---:|---:|---:|
| Seeds | `44001-44064` | `44065-44128` | 128 seeds |
| Games | 512 | 512 | 1,024 |
| Fallbacks | 0 | 0 | 0 |
| `verification.json valid` | `true` | `true` | `true` |
| `rilab_dirty` | `false` | `false` | `false` |
| Replay digest | `f2c39de8...88cb9fc` | `f910d917...a97bca3` | — |
| Protocol registration SHA-256 | `9b2adf0c...bf503ef3` | `3898ebe1...27d1e9eb4` | — |

- No anomalies, no unpaired first-vetoes.
- All 8 applied vetoes satisfy the registered scope: `opp_riichi_count = 1`,
  `turn_number` 13-17, v1.4.0 `riichi_discard`, v1.3.0 `discard`, and the
  executed action is `discard`.
- Treatment and control games are byte-identical except for the veto games:
  Stage 1 254/256 identical, Stage 2 252/256 identical. The differing games are
  exactly the games with an applied veto.
- `counts.json` for both stages has `outcomes_inspected: false`.

---

## 3. Count-only extension decision

| Stage | Treatment games | Veto events | Games with veto | First-veto pairs | Later vetoes |
|---|---:|---:|---:|---:|---:|
| Stage 1 | 256 | 2 | 2 | 2 | 0 |
| Stage 2 | 256 | 6 | 4 | 4 | 2 |
| Pooled | 512 | 8 | 6 | **6** | 2 |

Stage 1 first-veto pairs = **2 < 24**, so the pre-registered rule required
Stage 2. Stage 2 was run with the identical plan; pooled first-veto pairs =
**6 < 24**. No further extension is allowed.

The extension decision used counts only. No win/deal/income/loss value was read
before the Stage 2 decision.

---

## 4. Primary endpoint: first-veto hand deal rate

| Group | n pairs | Wins | Deals | Win rate | Deal rate | Income/hand | Loss/hand |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control (C) | 6 | 1 | 1 | 16.67% | **16.67%** | 1,717 | 333 |
| Treatment (T) | 6 | 1 | 1 | 16.67% | **16.67%** | 600 | 333 |

- Paired risk difference (T - C): **+0.00pp**
- Seed-cluster bootstrap 95% CI: **[+0.00pp, +0.00pp]**, 10,000 resamples,
  6 seed clusters
- Exact McNemar: **not defined** (0 discordant pairs: treatment-only 0,
  control-only 0, both deal 1, neither 5)
- Unpaired first-vetoes: **0**
- Unique hands: **6**

Pre-registered decision: **`underpowered`** (`n_pairs = 6 < 24` after the allowed
extension).

Because the decision is `underpowered`, no support/refutation statement is
allowed. The observed paired difference is exactly zero, but the sample is far
too small to distinguish a 10pp reduction from no effect.

### 4.1 Where the hand-level difference comes from

Both groups contain the same 6 paired hands:

- 1 pair: both deal (loss 2,000 each).
- 1 pair: both win; treatment income 3,600 versus control 10,300.
- 4 pairs: neither wins nor deals.

The income difference is therefore driven by one hand and is descriptive only.

---

## 5. Secondary and guardrails (exploratory, unadjusted)

| Endpoint | Treatment | Control | Difference | Bootstrap 95% CI |
|---|---:|---:|---:|---|
| Win rate | 16.67% | 16.67% | +0.00pp | [-50.00pp, +50.00pp] |
| Income/hand | 600 | 1,717 | -1,117 | [-5,150, +1,800] |
| Loss/hand | 333 | 333 | 0 | [0, 0] |

### 5.1 Game level and dilution check

| Arm | Games | Mean rank | 4th rate | Mean score |
|---|---:|---:|---:|---:|
| C | 512 | 2.4863 | 22.85% | 25,597.5 |
| T | 512 | 2.4883 | 22.85% | 25,596.9 |

- Paired game-level rank difference (T - C): `+0.0019`
  `[+0.0000, +0.0061]`
- Paired game-level score difference (T - C): `-0.4`
  `[-40.625, +29.102]`
- 4th-rate difference (T - C): `+0.00pp`
- Affected games: `6` of 512 treatment games (1.17%)

The all-game difference is negligible, as expected: only 6 games were affected.
This is the pre-registered dilution check, not evidence of no effect.

---

## 6. Trigger rate (post hoc, unverified diagnostic)

The only reproducible trigger count is the formal one: **8 applied vetoes in 512
treatment games, 6 games with at least one veto, and 6 first-veto pairs**
(`stage1/counts.json`, `stage2/counts.json`, `pooled/report.json`). This is the
count used for the extension decision and for the "operationally narrow"
statement in section 7.

I also looked post hoc at why the trigger was so rare. That look is **not** part
of the pre-registered analysis, is **not** a shadow replay, and is **not**
directly comparable to W3e:

- W3e Stage 1 B: 45 focal Riichi states with `opp_riichi_late` and 14 where old
  Turn would discard. This is derived from W3e `events.jsonl`, which was
  produced by a state-based v1.3.0 Turn shadow replay.
- W5 Stage 1 C: 47 focal Riichi events under an approximate late-threat count
  from `games/*.mjai.jsonl`, and 2 applied vetoes from `counts.json`.

The W5 numbers came from an **ad hoc, deleted script**, not from a committed
analysis path:

- W5 stored decision state only for applied vetoes. The 47 is therefore a
  game-log event count, not a count of captured decision states. It approximates
  the decision-time condition using the focal player's own draw count and
  opponent `reach_accepted` events, and it was not checked against the live
  state for all 47 events.
- The 2 is the applied-veto count. It is exact for the states where the veto
  backend actually made a decision, but it is not an independent v1.3.0 replay
  over all 47 events.
- A separate 2-seed engineering probe (16 games, deleted) re-ran both services
  with full focal-turn states and replayed v1.3.0 on 5 captured late-threat
  states: 2 discard, 3 Riichi. That validates the serving path for those seeds,
  but it does not establish the 47/2 rate.

Because the denominator, the numerator, and the W3e comparison use different
methods and different seeds, the 47/2 comparison is recorded here only as an
**unverified ad hoc diagnostic**, not as evidence. The formal, reproducible
statement is the trigger count above; no claim about a stable W3e divergence
rate is made.

---

## 7. Decision and interpretation

1. The pre-registered decision is **`underpowered`**: 6 first-veto pairs after
   the allowed Stage 2 extension, below the 24 required.
2. No claim of benefit or harm is made. The observed deal rate is identical
   (1/6 versus 1/6), the paired risk difference is 0, and the bootstrap interval
   is degenerate at 0 because no pair was discordant.
3. The intervention is operationally narrow: 8 veto events in 512 treatment
   games (6 first-veto games). The game-level guardrail is essentially unchanged.
4. The low trigger rate means a future test of this veto would need either many
   more seeds or a broader trigger. Broadening the trigger after seeing these
   results would be a new, separately pre-registered experiment.
5. W5 does not change the W2 verdict (`insufficient`), the v1.3.0 default, or
   the v1.4.0 candidate status.

---

## 8. What not to conclude

- Do not conclude that the veto is safe or beneficial. The sample cannot support
  either.
- Do not conclude that the W3e risk concentration was false. W5 did not
  re-sample the same states; it found fewer opportunities.
- Do not tune the veto threshold, the seed range, or the endpoint using these
  outcomes.
- Do not treat the 6 affected games as representative of ranked play.

---

## 9. Artifacts

```text
outputs/w5_riichi_veto_20261118/
  smoke/                         # 16 engineering games, excluded
  stage1/
    protocol.registered.json
    native_config.json
    batch/verification.json
    batch/summary.json
    batch/manifest.json
    batch/games/*.mjai.jsonl
    batch/decisions/*.jsonl.gz
    batch/results/*.json
    counts.json                  # count-only, no outcomes
  stage2/                        # same layout
  pooled/
    report.json
    report.md
```

Formal result: `outputs/w5_riichi_veto_20261118/pooled/report.json` and
`pooled/report.md`.

---

## 10. Commands

```powershell
cd F:\Code\AI\train_study\riichi_lab

D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --plan configs\w5\w5_riichi_veto_plan.yaml `
  --stage smoke --phase all --output-root outputs\w5_riichi_veto_20261118

D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --plan configs\w5\w5_riichi_veto_plan.yaml `
  --stage stage1 --phase run --output-root outputs\w5_riichi_veto_20261118
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --plan configs\w5\w5_riichi_veto_plan.yaml `
  --stage stage1 --phase count --output-root outputs\w5_riichi_veto_20261118

# Count showed 2 < 24, so Stage 2 was required.
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --plan configs\w5\w5_riichi_veto_plan.yaml `
  --stage stage2 --phase run --output-root outputs\w5_riichi_veto_20261118
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --plan configs\w5\w5_riichi_veto_plan.yaml `
  --stage stage2 --phase count --output-root outputs\w5_riichi_veto_20261118

D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --plan configs\w5\w5_riichi_veto_plan.yaml `
  --stage pooled --phase analyze --output-root outputs\w5_riichi_veto_20261118
```

`pytest` at the implementation commit: **297 passed**.

---

## 11. Limitations

- Fixed v1.3.0 opponents, local CPU, `4p-red-half`, `default_tenhou`; not online
  ranked play.
- Only 6 paired first-veto hands; the primary endpoint is underpowered by the
  pre-registered rule.
- The veto changes the game trajectory; only the first-veto hand is paired.
  Later vetoes (2 events in one Stage 2 game) are descriptive.
- Secondary endpoints are exploratory and unadjusted.
- The trigger-rate comparison with W3e uses different seed ranges and is
  descriptive; it is not a formal between-experiment test.
- Hand outcomes are whole-hand outcomes, not the immediate result of the
  substituted discard.
