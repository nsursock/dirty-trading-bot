# Stage 0 Protocol — Validate The Validator

**Status:** gates locked (`protocol_version: 0.1.0`)  
**Machine-readable gates:** [`stage0_gates.yaml`](./stage0_gates.yaml)  
**Issues:** [#19](https://github.com/nsursock/dirty-trading-bot/issues/19) · [#20](https://github.com/nsursock/dirty-trading-bot/issues/20) · [#21](https://github.com/nsursock/dirty-trading-bot/issues/21) · [dirty-fin-reports#20](https://github.com/nsursock/dirty-fin-reports/issues/20)

## Goal

Rerun the already-working `GBM + planted AR` case through the evaluation
harness that will later be trusted on real data. The planted alpha is known to
be real. If this harness cannot confirm a stable edge here, the harness is
wrong or mismatched to sequential RL.

## Freeze rule

1. Define pass criteria **before** any Stage 0 experiment run.
2. Do **not** relax thresholds after seeing results.
3. A deliberate pre-run revision must bump `protocol_version`, update
   `freeze_date` / `freeze_commit`, and note why — never because a run failed.

## Validation method

**Default:** expanding walk-forward with purge and embargo on a single long
synthetic path per seed.

```text
|---- train ----| purge | embargo |---- test (OOS) ----|
```

- Training never sees bars at or after `test_start`.
- Purge covers feature / position overlap between train and test.
- Embargo adds a dead zone after purge before OOS evaluation.

**CPCV:** deferred for Stage 0. Combinatorial purged CV is only acceptable for
sequential HRL if temporal causality is proven; otherwise prefer this
walk-forward protocol.

**IID seed holdout** (`TEST_SEED_OFFSETS` / `VALID_SEED_OFFSETS`) remains a
sanity mode for Optuna and smoke tests. It is **not** the Stage 0 trusted
protocol.

## Aggregation

| Unit | Rule |
| --- | --- |
| Fold observation | one `(seed, fold)` pair |
| Seed positive | median OOS `total_return` across that seed's folds `> 0` |
| Primary metric | `upi` (Ulcer Performance Index / Martin) |
| Return metric | `total_return` |

## Locked gates

| Gate | Rule |
| --- | --- |
| Median OOS return | median OOS `total_return` across folds `> 0` |
| Folds positive | ≥ 60% of folds have OOS `total_return` `> 0` |
| Seeds positive | ≥ 60% of seeds are positive (see above) |
| IS→OOS retention | median `(OOS upi / IS upi)` ≥ 50% when IS upi `> 0` |
| OOS UPI | median OOS `upi` `> 0` |
| Fold dominance | `max(|fold return|) / sum(|fold return|)` ≤ 50% |

Exact operators and thresholds live only in `stage0_gates.yaml`.

## Pass / fail

Stage 0 **passes** only when every gate in `stage0_gates.yaml` passes.
On pass: freeze this validator methodology; do not retune splits or gates in
response to later inconvenient outcomes.

## Positive-control reference (PASSED / FROZEN)

**Claim:** HRL can **recover and exploit** planted AR alpha at the **calibration
positive-control** under the locked walk-forward protocol, with a mapped
signal-strength boundary at fixed κ.
(Not: “HRL merely confirms α,” and not: “weak-signal φ=0.35 is tradable.”)

Immutable benchmark metadata:

```text
protocol_version = 0.1.0
dirty-fin-reports = 0.0.4
dirty-mlx-ml      = v0.0.5
phi_calibration   = 0.70     # positive-control world in configs/stage0.yaml
phi_stage1_op     = 0.60     # Stage 1 operating point (safety margin)
phi_deepest_pass  = 0.55
phi_first_fail    = 0.50
boundary          = (0.50, 0.55]
ar_noise (kappa)  = 1.71
returns.freq      = 1h
manager_updates   = 50
folds             = 4
seeds             = 3
purge_bars        = 48
embargo_bars      = 48
gates             = roadmap/stage0_gates.yaml (unchanged — do not edit)
```

Canonical artifacts:

- Positive control: `reference_runs/stage0_positive_control` → `logs/20260820-195642-704279-2052-stage0-phi0.7-k1.71`
- Ladder PASS φ=0.60: `logs/20260820-202813-364829-3096-stage0-phi0.6-k1.71-phi0.6-k1.71`
- Ladder PASS φ=0.55: `logs/20260820-204008-719460-3712-stage0-phi0.55-k1.71-phi0.55-k1.71`
- Ladder FAIL φ=0.50: `logs/20260820-203254-195724-3339-stage0-phi0.5-k1.71-phi0.5-k1.71`
- Config: `configs/stage0.yaml`
- Issue: [#19](https://github.com/nsursock/dirty-trading-bot/issues/19) (closed)

Signal-strength ladder is **complete** — see [`final_plan.md`](./final_plan.md).
Do not refine φ further. Stage 1 uses **φ=0.60 / κ=1.71**. Later stages change
exactly one difficulty dimension vs this operating point.

**UPI note:** values ~10³ are expected at strong edge — UPI = (mean excess @ 1h)×6048 /
Ulcer with Ulcer ~10⁻³. Near the boundary, prefer seed/return/fold breadth over
UPI when coverage < ~80%; partial-coverage retention can be a survivorship artifact.

## How to run

```bash
caffeinate -dims venv/bin/python scripts/stage0.py \
  --config configs/stage0.yaml \
  --gates roadmap/stage0_gates.yaml
```

Smoke harness only:

```bash
caffeinate -dims venv/bin/python scripts/stage0.py \
  --config configs/stage0_smoke.yaml \
  --gates roadmap/stage0_gates.yaml
```

Artifacts land under `logs/<run>/stage0/` including `stage0_report.json`.
