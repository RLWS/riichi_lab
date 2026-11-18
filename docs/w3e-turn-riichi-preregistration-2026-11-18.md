# W3e pre-registration: Turn-only replication of the "risky extra riichi" hypothesis (2026-11-18)

- Status: **pre-registered before any W3e game was run or any W3e outcome was inspected**
- Branch: `codex/w3e-turn-riichi`, base commit `2d643c5341430fcb4496156cdd45184846203b9a`
- Plan file: `configs/w3e/w3e_turn_riichi_plan.yaml`
- Runner: `scripts/run_w3e_turn_riichi.py`
- Analysis module: `rilab/w3e.py`
- Output root (planned): `outputs/w3e_turn_20261118/`
- Nature: **descriptive / association replication, not a causal counterfactual experiment**

---

## 0. One-paragraph summary

W3d reported a small-sample descriptive signal: in `riichi_selected` states where the
new v1.4.0 Turn declared riichi but the old v1.3.0 Turn would have discarded, the hands
won less and dealt in more than the mirror states (old Turn declared riichi, new Turn
would discard). W3e independently replicates or rejects that signal with new seeds,
Turn-only arms, and a fixed old Response. It does **not** retrain, recalibrate, change
chi/pon/riichi penalties, or reuse W2 formal seeds. The W2 formal verdict remains
`insufficient`; v1.3.0 remains the default and v1.4.0 remains a candidate regardless of
the W3e result.

---

## 1. Hypothesis under test

**H1 (direction).** Executed Turn riichi hands that exist only because the new Turn
declares riichi where the old Turn would discard (`B_extra`) have a higher hand-level
deal rate than the mirror executed riichi hands that exist only because the old Turn
declares riichi where the new Turn would discard (`A_mirror`).

W3d's descriptive estimate was 37.5% deal for B (`n=8` hands) versus 6.25% for the A
mirror (`n=16` hands), with the B group concentrated in opponent-riichi / late-turn
states. W3e asks whether that direction survives a new, larger, pre-registered sample.

**H2 (secondary, risk concentration).** `B_extra` hands contain a larger share of
`opp_riichi_late` states (opponent riichi already accepted and `turn_number > 12`) than
`A_mirror` hands.

W3e can replicate or reject an **association between declaration-threshold divergence
and hand outcome**. It cannot prove that the extra declaration caused the deal, because
`B_extra` and `A_mirror` are different states in different games and are not paired
counterfactuals. Even a "replicated" verdict is worded as a replicated association.

---

## 2. Design

### 2.1 Turn-only A/B replication

| Arm | Turn | Response | Purpose |
|---|---|---|---|
| A | v1.3.0 (`full_state_v2`) | v1.3.0 (`full_state_v2`, `pass_bias=1.5`) | all-old control |
| B | v1.4.0 (`full_state_v3`) | v1.3.0 (`full_state_v2`, `pass_bias=1.5`) | new Turn only |

Response is fixed to old v1.3.0 in both arms so that the only changed component is
Turn. C and D are not run in W3e. The three opponents are fixed v1.3.0 services, as in
W2. No model weights, calibration, `pass_bias`, `action_biases`, or
`no_yaku_call_penalty` are changed.

### 2.2 Engine and schedule

- Game mode: `4p-red-half`
- Rule: `default_tenhou`
- Four seat rotations per seed and arm; the focal seat is the rotated seat.
- A arm seats: `["v130", "v130", "v130", "v130"]`.
- B arm seats: `["B", "v130", "v130", "v130"]` with the B seat rotated to each of the
  four seats.
- Games per seed: 2 arms × 4 rotations = 8.
- Stage 1: 64 seeds × 8 = **512 games**.
- Stage 2 (only if the extension rule fires): 64 seeds × 8 = **512 games**.
- Smoke (engineering only): 2 seeds × 8 = **16 games**.

### 2.3 Seeds

| Purpose | Seeds | Count | Evidence use |
|---|---|---:|---|
| Smoke | `43501-43502` | 2 | engineering only; excluded from formal statistics |
| Stage 1 | `43001-43064` | 64 | formal W3e sample |
| Stage 2 (conditional) | `43065-43128` | 64 | formal extension sample |

Hard exclusions, checked by the validator:

- `40001-40008` A3 engineering seeds
- `41001-41128` reserved W1 seeds
- `42001-42128` W2 formal seeds

`43001-43128` was verified unused in the repository before registration. Smoke seeds
are disjoint from both formal stages and from all reserved ranges.

### 2.4 State capture and same-state shadow replay

The W2 decision spool stores full `state` only every 100 decisions, which is not enough
for a per-hand divergence endpoint. W3e therefore adds an optional capture filter to
`rilab.batch.run_batch`:

- Capture the full decision record only when:
  - `player_id == focal_seat`, and
  - `task == "turn"`, and
  - the executed `selected_action.kind == "riichi_discard"`.
- All other decision records keep the existing compact form with `state` stripped.
- Existing W2 behavior is unchanged when no capture filter is supplied.

For every captured executed riichi declaration state, both Turn services are replayed
on the **same state and candidate list**:

- Arm B: replay old v1.3.0 Turn; `B_extra` if old selects `discard`.
- Arm A: replay new v1.4.0 Turn; `A_mirror` if new selects `discard`.
- If both select `riichi_discard`, the state is a `both_riichi` state.
- If the other version selects anything else (`tsumo`, `ankan`, `kakan`), the state is
  recorded but is not part of `B_extra`/`A_mirror`.
- States whose candidate count exceeds the common service support
  (`min(v130.candidate_limit, v140.candidate_limit)`, currently 64) are excluded from
  shadow classification and reported as `excluded_common_support`.
- A shadow replay error invalidates the stage; it is not silently treated as a
  non-divergence.

The full state is normalized exactly as in W3b/W3c/W3d: `full_state_v2` for v1.3.0,
`full_state_v3` for v1.4.0. The service metadata is checked unchanged before and after.

### 2.5 Unit of analysis and deduplication

- A hand can declare riichi at most once, so each captured declaration normally maps to
  one unique hand.
- Defensively, the analysis deduplicates by `(game_id, hand_id)` and keeps the lowest
  `decision_id` if duplicates ever appear.
- Primary and secondary outcome rates are computed over **unique hands**, not decision
  rows.
- The per-seed view groups hands by the seed of their game. Because A and B games share
  a seed but are separate trajectories, seed-level pairing is a sensitivity analysis
  only, not the primary unit.

---

## 3. Endpoints and analysis

### 3.1 Primary endpoint

Metric: **hand-level deal rate** of executed riichi hands in `B_extra` versus
`A_mirror`.

Definitions:

- `B_extra`: arm B, executed `riichi_discard`, old v1.3.0 Turn shadow selects
  `discard` on the same state.
- `A_mirror`: arm A, executed `riichi_discard`, new v1.4.0 Turn shadow selects
  `discard` on the same state.
- Deal: the focal player is the dealt-in player (`target == focal`) on that hand, as
  computed by `rilab.w3.hand_metrics`.
- Win: the focal player wins that hand.

Reported quantities:

- `n_unique_hands`, wins, deals
- deal rate with Wilson 95% score interval for each group
- risk difference `p_B_extra - p_A_mirror`
- Newcombe 95% hybrid-score interval for the difference
- two-sided Fisher exact p-value (descriptive; no multiplicity correction)
- seed-cluster bootstrap 95% interval for the difference (10,000 resamples; sensitivity)
- the same quantities for win rate and income/loss per hand as secondary outcomes

### 3.2 Pre-registered primary decision rule

Let `d = p_B_extra - p_A_mirror`, `L/U` the Newcombe 95% interval, and
`n_min = 24` unique hands in each group.

- **`replicated`**: `n_B_extra >= 24`, `n_A_mirror >= 24`, `L > 0`, and `d >= +0.10`.
- **`rejected`**: `n_B_extra >= 24`, `n_A_mirror >= 24`, and `U <= +0.10`.
  This means the data rule out a deal-rate increase of 10 percentage points or more;
  it is the pre-registered rejection of the risky-extra-riichi hypothesis.
- **`underpowered`**: fewer than 24 unique hands in either group after the extension
  rule has been applied and any allowed Stage 2 has run.
- **`inconclusive`**: all other cases (including `L > 0` with `d < +0.10`, or an
  interval that contains both 0 and +0.10).

No other threshold may be substituted after seeing results. A `replicated` verdict is
still an association, not proof that the declaration caused the deal.

### 3.3 Extension rule (decided before outcomes)

After the Stage 1 batch has been audited and the same-state shadow classification has
been run, count **only** the number of unique `B_extra` hands. Do **not** compute or
inspect win/deal/income/loss before this decision.

- If `B_extra unique hands < 24`, run Stage 2 (`43065-43128`) with the identical plan,
  arms, rotations, capture filter, services, and analysis.
- If `B_extra unique hands >= 24`, do not run Stage 2; analyze Stage 1 only.
- If Stage 2 runs, the **pooled** Stage 1+2 sample is the primary analysis. The
  Stage 1-only table is reported as the pre-planned interim result.
- The count phase writes `counts.json` with event counts and **no outcome fields**.
  The analyze phase is a separate command, so the extension decision cannot be driven
  by outcome rates.
- No outcome-based early stopping, no seed replacement, no optional re-run based on
  results. An engineering failure (service error, invalid audit) stops the stage and is
  fixed before rerunning the same seeds; the failure is disclosed.

Sample-size projection from W3d, **not an outcome and not a guarantee**: W2 B had about
2.2 focal riichi hands per B game (about 8.8 per four-rotation seed), and W3d's sampled
divergence rate was 8/106 ≈ 7.5%. Stage 1 (256 B games) therefore projects to roughly
40 `B_extra` hands and roughly 95 `A_mirror` hands. The 24-hand threshold is a minimum
precision gate, not a power calculation.

### 3.4 Secondary endpoints (all exploratory, unadjusted)

1. Win rate, income per hand, loss per hand for `B_extra` versus `A_mirror`.
2. `opp_riichi_late` subset (`opp_riichi_count > 0` and `turn_number > 12`):
   deal rate for `B_extra` versus `A_mirror`; share of each group that is
   `opp_riichi_late`; risk difference with Newcombe 95% interval.
3. W3d hand-shape metrics for the executed riichi discard:
   tenpai rate, wait-tile count, remaining waits, max han, good-wait and bad-wait
   rates, for `B_extra`, `A_mirror`, `B_both_riichi`, and `A_both_riichi`.
   `remaining` uses the W3d visibility rule (hand, all discards, all melds, dora
   indicators; red fives counted as normal fives). `max_han` uses actual dora plus
   riichi and does not include ura dora.
4. The other version's exact shadow riichi shape when it also selects
   `riichi_discard`; paired same-tile rate and `Δremaining` / `Δmax_han` versus the
   executed discard.
5. Both-riichi discard agreement: among states where both versions select
   `riichi_discard`, the fraction with the same discard tile.
6. Context comparison `B_extra` versus `B_both_riichi` (within-arm reference) for deal
   rate and shape. This is reported but does not replace the primary endpoint.
7. `B_extra` vs `A_mirror` split by `opp_riichi` and by `late`, with counts and Wilson
   intervals. These cells may be too small to interpret.

Secondary intervals are unadjusted and are not used to declare a confirmed defect.

### 3.5 Statistical helpers

- Wilson score interval, `z = 1.959963984540054`.
- Newcombe hybrid-score interval for the difference of two independent proportions
  (method 10).
- Two-sided Fisher exact test, implemented with exact hypergeometric probabilities
  (`math.comb`), no SciPy dependency.
- Cluster bootstrap by seed: resample seeds with replacement; within each resampled
  seed pool that seed's hands; skip resamples where either group has no hand; report
  the number of valid resamples. Fixed seed `20261118`, 10,000 resamples.
- No multiplicity correction. The primary endpoint is single and pre-registered; all
  other endpoints are descriptive.

---

## 4. Run-integrity gates

A stage is analyzable only if all of the following hold:

- `run_batch` summary `valid == true`
- `completed_games == expected_games`
- `fallbacks == 0` across all games
- `audit_batch(stage_dir)` returns exactly the completed games
- every game is marked `valid`
- `verification.json` is written and `valid == true`
- shadow replay errors == 0
- every service `check_unchanged()` passes before and after replay

If any gate fails, the stage is invalid. Do not analyze outcomes, do not replace seeds,
and do not relax the gate. Fix the engineering failure and rerun the same stage with
the same seeds, documenting the failure and rerun.

Stop conditions:

- missing service checkpoint/config (report the exact missing path; do not invent it)
- service startup failure or model metadata mismatch
- any fallback, illegal action, or audit mismatch
- shadow replay error or common-support exclusion that removes a material share of
  captured states (report the count; do not redefine the endpoint)
- Stage 1 count below 24 after Stage 2: mark `underpowered`, do not claim replication
  or rejection.

---

## 5. What W3e does not do

- It does not retrain, fine-tune, or recalibrate any model.
- It does not change chi/pon/riichi penalties or `pass_bias`.
- It does not run arms C or D; it isolates Turn against a fixed old Response.
- It does not reuse W2 formal seeds `42001-42128`, W1 seeds `41001-41128`, or A3 seeds
  `40001-40008`.
- It does not change the W2 formal verdict (`insufficient`), the v1.3.0 default, or the
  v1.4.0 candidate status.
- It does not treat the shadow action as executed. `A_mirror` and `B_extra` are
  different executed trajectories; the shadow only classifies which states belong to
  the divergence group.
- It does not estimate a full EV, push/fold threshold, or online ranked strength.

---

## 6. Limitations fixed in advance

1. **Between-game comparison, not paired counterfactual.** `B_extra` and `A_mirror`
   occur in different games and different states. The mirror is a comparison group, not
   the same hand under the other policy.
2. **State-mix confounding.** If `B_extra` contains more late/opponent-riichi states,
   the deal-rate difference may reflect state selection rather than declaration quality.
   H2 measures the mix, but does not remove the confounding.
3. **Small divergence cells.** W3d's W2 sample had 8 and 16 hands; W3e projects ~40 and
   ~95, but the `opp_riichi_late` cells will remain small.
4. **Hand-level outcome, not immediate deal tile.** The deal is the final outcome of the
   hand, not necessarily the tile discarded at the divergence state.
5. **Fixed local opponents and rules.** Three fixed v1.3.0 opponents, local CPU,
   `4p-red-half` / `default_tenhou`; not online ranked play.
6. **Old Response everywhere.** W3e says nothing about Response; the W2 D arm mixed
   Turn and Response and cannot be used as W3e evidence.
7. **Descriptive shadow classification.** The service replay is one-step; it does not
   continue the game under the shadow action.
8. **No ura dora, riichi sticks, or full situational EV** in the shape metrics.
9. **Multiple secondary cells.** Unadjusted intervals; do not promote a secondary cell
   to a confirmed finding.
10. **W3e is not a release experiment.** Its result does not by itself change model
    status. Any release decision still needs the W2/W1 evidence path.

---

## 7. Planned commands and artifacts

```powershell
cd F:\Code\AI\train_study\riichi_lab

# 0) dry-run: schedule and seed validation only
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage1 --phase run --dry-run

# 1) smoke (engineering only, excluded from formal statistics)
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage smoke --phase all --output-root outputs\w3e_turn_20261118

# 2) Stage 1 batch + count-only shadow classification (no outcomes)
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage1 --phase run --output-root outputs\w3e_turn_20261118
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage1 --phase count --output-root outputs\w3e_turn_20261118

# 3) extension decision uses counts.json only; if <24 B_extra hands:
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage2 --phase run --output-root outputs\w3e_turn_20261118
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage2 --phase count --output-root outputs\w3e_turn_20261118

# 4) outcome analysis (Stage 1 only, or pooled if Stage 2 ran)
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage stage1 --phase analyze --output-root outputs\w3e_turn_20261118
# pooled only after the Stage 2 extension:
D:\Anaconda3\python.exe scripts\run_w3e_turn_riichi.py `
  --plan configs\w3e\w3e_turn_riichi_plan.yaml `
  --stage pooled --phase analyze --output-root outputs\w3e_turn_20261118
```

Artifacts per stage:

```text
outputs/w3e_turn_20261118/<stage>/
  protocol.registered.json     # written before the first game
  native_config.json
  services/                    # service logs + resolved configs
  batch/
    manifest.json
    summary.json
    verification.json
    games/*.mjai.jsonl
    decisions/*.jsonl.gz      # full state only for captured Turn riichi states
    results/*.json
  events.jsonl                # captured states + same-state shadow actions + shape; no outcomes
  counts.json                 # event counts only; no outcomes
  hands.jsonl                 # analyze phase: events joined to hand outcomes, deduped
  report.json
  report.md
```

---

## 8. Registration checksum and change control

The YAML plan is the machine-readable registration. `protocol.registered.json` records
the plan SHA-256, service-config SHA-256 values, arm definitions, seed list, schedule
SHA-256, capture policy, analysis rules, and stopping rules before the first game.

Any change to seeds, arms, endpoints, thresholds, extension rule, or capture policy
after this document is written must be recorded as a dated deviation **before** the
affected analysis, with the reason and the resulting plan hash. Results computed under
the old plan remain reported under the old hash.

---

## 9. Decision and reporting language

- If `replicated`: report "the W3d direction replicated as an association under the
  pre-registered W3e comparison"; explicitly state that this is not a causal proof and
  does not change the W2 `insufficient` verdict by itself.
- If `rejected`: report "the pre-registered W3e data rule out a >=10pp deal-rate
  increase in the hypothesised direction"; do not say "the new Turn is safe" in
  general.
- If `inconclusive` or `underpowered`: report exactly that; do not reinterpret a
  favorable or unfavorable point estimate as confirmation.
- Never write "confirmed defect", "confirmed regression", "v1.4.0 is stronger", or
  "v1.4.0 is weaker" from W3e alone.
