# W5 pre-registration: serving-time veto of extra late Riichi under threat

- Status: **pre-registered before any W5 game is played**
- Date: 2026-11-18
- Branch: `codex/w5-riichi-veto`
- Base commit: `858e4f0` (`codex/v1.4.0-w2`, includes W3e)
- Plan: `configs/w5/w5_riichi_veto_plan.yaml`
- Runner: `scripts/run_w5_riichi_veto.py`
- Backend: `rilab/backends/veto_turn.py`

This is a **serving-time intervention**, not a retraining experiment. No model
weights, training data, calibration values, or global chi/pon/riichi penalties
are changed.

---

## 1. Motivation and scope

W3e Stage 1 replicated the W3d direction: the v1.4.0 Turn model declares Riichi
where v1.3.0 Turn would discard (`B_extra`), and those `B_extra` hands dealt
32.35% versus 14.81% for the mirrored old-Turn extra Riichi (`A_mirror`),
+17.54pp, Newcombe CI [+1.36, +35.43], Fisher p=0.0418, seed-cluster bootstrap
CI [-0.12, +35.96]. The effect was concentrated in `opp_riichi_late`:
14 `B_extra` hands, 0 wins, 42.86% deal rate, 4,329 loss per hand. The other 20
`B_extra` hands had a 50% win rate, 25% deal rate, and positive income.

W5 asks a narrow, practical question:

> If v1.4.0 Turn is kept everywhere except that its `riichi_discard` under
> `opp_riichi_count > 0` and `turn_number > 12` is replaced by v1.3.0 Turn's
> `discard` on the same state, do the affected hands deal less often?

This tests whether the W3e risk concentration is actionable at serving time. It
does not test a retrained policy, a general Riichi threshold, or a new Response
model.

The W2 verdict remains `insufficient`. v1.3.0 remains the default deployment and
v1.4.0 remains a candidate. W5 cannot change either status by itself.

---

## 2. Arms

Both arms use v1.4.0 Turn and v1.3.0 Response.

| Arm | Turn | Veto | Response |
|---|---|---|---|
| `C` control | v1.4.0 | none | v1.3.0 |
| `T` treatment | v1.4.0 | old-v1.3.0 discard substitution | v1.3.0 |

- Opponents: three fixed v1.3.0 seats.
- Seats: `C` = `["C","v130","v130","v130"]`, `T` = `["T","v130","v130","v130"]`,
  rotated through all four positions.
- Focal seat: the rotation index (the seat occupied by `C` or `T`).
- Response is fixed to v1.3.0 in both arms. There is no Response contrast.
- The v1.3.0 veto service is the same pinned service used by the old Turn.

### 2.1 Veto rule (exact)

On a `turn` decision, after v1.4.0 Turn returns its prediction:

1. If the selected kind is not `riichi_discard`, keep v1.4.0 unchanged.
2. If `opp_riichi_count <= 0`, keep v1.4.0 unchanged.
3. If `turn_number <= 12`, keep v1.4.0 unchanged.
4. Otherwise call v1.3.0 Turn on the same legal candidates with a v2-normalized
   state copy: `feature_schema = "full_state_v2"` and `own_discards` removed.
5. If v1.3.0 selects `discard`, replace the v1.4.0 `riichi_discard` with that
   `discard` and record veto metadata.
6. If v1.3.0 selects anything else, including `riichi_discard`, keep v1.4.0
   unchanged.

`opp_riichi_count` is the number of opponents whose Riichi has been accepted,
computed from `riichi_declared_by_player` with the acting seat excluded. This is
the same definition used by W3e `state_features`.

The veto is applied only in the `T` arm. The `C` arm has no veto backend.

Veto metadata is stored under `prediction.veto`:

```json
{
  "applied": true,
  "reason": "opp_riichi_late_riichi_to_discard",
  "new_selected": {"kind": "riichi_discard", "tile": "..."},
  "old_selected": {"kind": "discard", "tile": "..."},
  "old_model": {"task": "turn", "...": "..."},
  "old_predictions": {"task": "turn", "selected_action": {"...": "..."}, "...": "..."}
}
```

`new_selected` is the original v1.4.0 action. `old_selected` is the replacement
v1.3.0 discard. `old_predictions` is the validated v1.3.0 Turn response used for
the decision.

---

## 3. Seeds and batch size

- Smoke (engineering only): `44501-44502`.
- Stage 1: `44001-44064` (64 seeds).
- Stage 2: `44065-44128` (64 seeds).
- Stage 1: 64 seeds x 2 arms x 4 rotations = **512 games**.
- Stage 2 (only if the count rule requires it): 64 seeds x 2 arms x 4 rotations
  = **512 games**.
- Smoke seeds are excluded from all statistics.

Forbidden seed ranges, already used by earlier experiments:

- `40001-40008` (A3 engineering)
- `41001-41128` (W1)
- `42001-42128` (W2)
- `43001-43128` (W3e)

The plan validates that smoke, Stage 1 and Stage 2 are unique, pairwise
disjoint, and disjoint from every forbidden range. Seeds were also checked
against the repository before registration; `44001-44128` and `44501-44502`
were unused.

---

## 4. Primary endpoint and pairing

**Primary endpoint:** hand-level deal rate of the **first-veto hand**, treatment
versus control, paired by `(seed, rotation, hand_id)`.

- Only the **first** veto per treatment game is used for the primary paired
  analysis. Later vetoes in the same game are descriptive only.
- The control game is the `C` game with the same seed and rotation.
- Because both arms use the same v1.4.0 Turn and v1.3.0 Response, same seed and
  same rotation, the two games are identical up to the first veto decision.
  The comparison is therefore a paired intervention at that decision.
- The hand outcome is the full-hand outcome from `rilab.w3.hand_metrics` for the
  focal seat: win, deal, income, loss.
- The hand id comes from the treatment decision record. The same hand id is
  looked up in the control game log.

Pair validity checks:

1. The control game must exist in the same stage batch.
2. The control decision log must exist.
3. Every decision before the first-veto `decision_id` must match between the
   treatment and control games on `player_id`, `task`, `hand_id`,
   `selected_action`, `engine_action`, and `new_events`.
4. The control record at the veto `decision_id` must have the same `player_id`,
   `task`, and `hand_id`.
5. The hand id must index a hand in both game logs.

If any check fails, the pair is marked `unpaired` with a reason and reported.
Unpaired first-vetoes are never silently dropped.

---

## 5. Count-only phase and extension rule

The `count` phase reads decision artifacts and writes **only** counts and IDs.
It does not read game logs, hand outcomes, win/deal rates, or income/loss.

Count field: `first_veto_pairs`, the number of treatment games with at least one
applied veto whose control game and control decision log exist.

- If Stage 1 `first_veto_pairs < 24`: run Stage 2 with the identical plan and
  analyze the pooled sample.
- If Stage 1 `first_veto_pairs >= 24`: do **not** run Stage 2; analyze Stage 1.

There is no outcome-based early stopping and no optional extension. The
extension decision is based only on `counts.json`.

---

## 6. Analysis

### 6.1 Primary

For the paired first-veto hands:

- Treatment deal rate and control deal rate.
- Paired risk difference `RD = p_T - p_C`, which equals
  `(b - c) / n` where `b` is treatment-deal/control-safe and `c` is
  treatment-safe/control-deal.
- Exact two-sided McNemar test on `(b, c)`.
- Paired 95% CI from a seed-cluster bootstrap: resample seeds with replacement,
  collect all paired hands in the drawn seeds, compute the paired risk
  difference, 10,000 resamples, percentile 2.5/97.5.
- Report `n_pairs`, unique hands, wins, deals, income, loss, and the discordant
  table.

### 6.2 Decision rule (pre-registered)

- `underpowered` if `n_pairs < 24` after the allowed extension.
- `supported` if `n_pairs >= 24`, the treatment deal rate is lower than control
  by at least 10 percentage points (`RD <= -0.10`), and the paired 95% CI upper
  bound is below 0.
- `not_supported` if `n_pairs >= 24` and the estimate does not meet the support
  rule.

The decision uses the seed-cluster bootstrap interval, not the McNemar p-value.
The McNemar p-value is reported for transparency.

### 6.3 Secondary and guardrails (exploratory, unadjusted)

- First-veto hand win rate, income per hand, loss per hand.
- Game-level rank and score difference, paired by `(seed, rotation)`, across all
  games.
- All-game mean rank and 4th rate for both arms (dilution check).
- Number of veto events, games with at least one veto, later vetoes, and the
  affected-hand subset of game-level outcomes.
- Unpaired first-veto list and reasons.

Secondary endpoints are exploratory and unadjusted. They cannot upgrade or
downgrade the primary decision.

---

## 7. Implementation contract

- `VetoTurnBackend` (`rilab/backends/veto_turn.py`) is registered as
  `backend_type: "veto_turn"`.
- Constructor config: `turn_backend` (v1.4.0), `veto_backend` (v1.3.0),
  `response_backend` (v1.3.0).
- `predict("response", ...)` routes to the v1.3.0 Response backend.
- `predict("turn", ...)` routes to the v1.4.0 Turn backend and applies the veto
  rule above.
- The v1.3.0 Turn call uses `rilab.w3_shadow.normalize_state(state, "full_state_v2")`.
- `metadata`, `calibration`, `requests`, `profile_for_task`, `create_adapter`,
  `validate_adapter`, `check_unchanged`, and `close` follow the existing
  task-routed backend contract.
- `create_adapter` returns `TaskRoutedAdapter(v3 turn, v2 response)`.

Artifacts:

- Full state is retained only for focal-seat turn decisions where the veto was
  actually applied. All other compact records keep the normal state-stripped
  form.
- `counts.json` has `outcomes_inspected: false` and contains IDs only.
- `report.json` / `report.md` are written only by the `analyze` phase.

---

## 8. Non-goals and limitations

- No retraining, no checkpoint change, no calibration change.
- No global Riichi/chi/pon penalty tuning.
- Fixed v1.3.0 opponents, local CPU, `4p-red-half`, `default_tenhou`; not online
  ranked play.
- The veto is a serving-time substitution and may create a different game
  trajectory; the first-veto hand is paired at the decision, but later hands
  are not paired counterfactuals.
- The expected number of `opp_riichi_late` first-vetoes is small. The
  pre-registered extension rule exists because Stage 1 alone may be
  underpowered.
- Secondary endpoints are exploratory and unadjusted.
- W5 cannot establish that v1.4.0 Turn is unsafe in general. It tests only the
  narrow late-threat extra-Riichi subset.

---

## 9. Commands (to be run after this document and plan are committed)

```powershell
cd F:\Code\AI\train_study\riichi_lab

# Dry run
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py --stage stage1 --phase run --dry-run

# Engineering smoke (2 seeds, 16 games)
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --stage smoke --phase all --output-root outputs\w5_riichi_veto_20261118

# Stage 1 batch
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --stage stage1 --phase run --output-root outputs\w5_riichi_veto_20261118

# Count-only extension decision
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --stage stage1 --phase count --output-root outputs\w5_riichi_veto_20261118

# Stage 2 only if Stage 1 first_veto_pairs < 24
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --stage stage2 --phase run --output-root outputs\w5_riichi_veto_20261118
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --stage stage2 --phase count --output-root outputs\w5_riichi_veto_20261118

# Outcome analysis: Stage 1 if no extension, pooled if extended
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --stage stage1 --phase analyze --output-root outputs\w5_riichi_veto_20261118
# or
D:\Anaconda3\python.exe scripts\run_w5_riichi_veto.py `
  --stage pooled --phase analyze --output-root outputs\w5_riichi_veto_20261118
```

No plan, seed, arm, veto rule, endpoint, or threshold may be changed after the
Stage 1 batch starts.
