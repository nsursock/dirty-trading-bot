# Final Roadmap Plan

## Core Objective

The goal is not merely to make HRL profitable on synthetic data. That has already been shown on `GBM + planted AR`.

The real goal is to demonstrate that the system can recover a known weak signal under progressively harder conditions, and that the edge survives strict out-of-sample validation under controlled distribution shift.

This implies four rules:

1. Increase only one difficulty dimension at a time.
2. Keep ground truth known for as long as possible.
3. Insert validation gates early, not only at the end.
4. Separate three distinct questions:
   - Is there alpha?
   - What regime am I in?
   - What policy should I trade?

## Stage 0: Validate The Validator

**Status: PASSED / FROZEN.** Do not refine φ further (no 0.525). See
[`stage0_protocol.md`](./stage0_protocol.md) for immutable metadata.

Before making the market harder, rerun the already-working `GBM + AR` case through the same walk-forward or CPCV-style protocol that will later be trusted on real data.

This must come first because the planted alpha is known to be real. If the evaluation harness cannot confirm a stable edge here, then either the harness is wrong, or it is mismatched to the sequential RL setting.

Gate:

- The known planted alpha must survive the intended evaluation harness.

**Result:** HRL recovered and exploited planted α under locked gates
`protocol_version=0.1.0`. Calibration positive control is `φ=0.70` / `κ=1.71`.
The signal-strength ladder below is **complete**.

```text
κ = 1.71 fixed
Stage-1 operating φ     = 0.60   (0.05 above deepest PASS)
Deepest passing rung    = 0.55
First failing rung      = 0.50
Empirical boundary      = (0.50, 0.55]
```

**Interpretation:** At κ=1.71, recoverability persists through φ=0.55 and breaks
between φ=0.50 and φ=0.55, with **seed-level agreement** the clearest transition
signal. φ=0.60 is the Stage 1 operating point — not φ=0.55 (avoid operating on
the observed boundary).

### Stage 0 Numeric Gates

**Locked protocol:** [`stage0_protocol.md`](./stage0_protocol.md) + [`stage0_gates.yaml`](./stage0_gates.yaml) (`protocol_version: 0.1.0`).

Before running Stage 0, define the pass criteria up front and do not relax them after seeing results.

Locked initial thresholds:

- median out-of-sample return across folds must be positive
- at least 60% of folds must be positive
- at least 60% of seeds must be positive
- out-of-sample performance must retain at least 50% of in-sample performance on the primary metric
- UPI or Martin ratio must remain positive out of sample
- no single fold should dominate the result

These numbers can be revised later, but only before the experiment is run. Once Stage 0 passes, freeze the validator and stop adjusting the validation methodology in response to inconvenient later outcomes.

**Near-boundary evidence hierarchy** (read order when gates disagree / coverage thins):

1. Seed agreement  
2. Median OOS return  
3. Fold positivity  
4. UPI / risk-adjusted diagnostics (only when coverage is high)  
5. Fold dominance  

**Known gate caveats (do not edit `stage0_gates.yaml`):**

- **Fold dominance — low value in this regime.** Values stayed ~0.16–0.22 on both
  φ=0.70 PASS and φ=0.50 FAIL. Flagged for a future *pre-run* protocol bump if a
  later stage needs a discriminating concentration gate; not retired mid-stream.
- **UPI under partial coverage — survivorship risk.** Retention/OOS-UPI can PASS
  on the defined subset while breadth gates FAIL (seen at φ=0.50, cov≈50%).
  Future work (new protocol version): coverage-weighted UPI and/or a stricter
  evidence floor before UPI counts as supporting evidence. Existing min-coverage
  guards in dirty-fin-reports 0.0.4 remain; they prevent `None`-driven medians
  but do not remove conditional-on-survivors bias.

Runner:

```bash
caffeinate -dims venv/bin/python scripts/stage0.py \
  --config configs/stage0.yaml \
  --gates roadmap/stage0_gates.yaml
```

### Signal-strength ladder (COMPLETE — frozen)

Not roadmap Stage 1. Controlled difficulty probe on the Stage 0 positive
control: **hold `κ=1.71` fixed, change only `φ`**, reuse locked gates. Do not
retune `ar_noise` per step. **No further φ steps** (including 0.525).

| Step | φ | κ | Role |
|------|---|---|------|
| Stage 0 | 0.70 | 1.71 | Calibration positive control — **PASSED** (12/12, 3/3) |
| 0-φ.60 | 0.60 | 1.71 | **PASSED** (10/12, 3/3) — Stage 1 operating point |
| 0-φ.55 | 0.55 | 1.71 | **PASSED** (9/12, 3/3) — safety margin for 0.60 |
| 0-φ.50 | 0.50 | 1.71 | **FAILED** (6/12, **1/3 seeds**) — regime break |
| ≤0.40 | ≤0.40 | 1.71 | Skip / known FAIL |

```text
φ=0.70  PASS  ████████████  3/3 seeds
φ=0.60  PASS  ██████████░░  3/3 seeds
φ=0.55  PASS  █████████░░░  3/3 seeds  ← deepest PASS
φ=0.50  FAIL  ██████░░░░░░  1/3 seeds  ← regime break
```

Next difficulty axis (later, not Stage 1): raise κ with φ fixed. Do not mix axes.

## Stage 1: Baseline Synthetic World

**Operating world:** `φ=0.60`, `κ=1.71` (safety margin above deepest PASS 0.55).
Reuse Stage 0 locked gates / walk-forward protocol. This is **not** live portfolio
construction — it is the controlled synthetic ablation:

Prerequisite: Stage 0 **PASSED / FROZEN** (ladder complete).

Start from the simplest controlled lab:

- GBM base
- stationary planted AR alpha
- controlled Gaussian noise
- no regimes
- no stochastic volatility
- no fat tails
- no frictions
- no multi-asset structure

Run three variants:

- `HRL only`
- `true alpha -> HRL`
- `predicted alpha -> HRL`

This establishes whether an explicit alpha layer helps over raw HRL, and how far the system is from the oracle ceiling.

Gate:

- `true alpha >= predicted alpha >= raw HRL` in out-of-sample behavior.

## Stage 2: Alpha Detector First

Do not add regime modeling yet.

Train a supervised alpha detector on the synthetic world.

Inputs:

- rolling returns
- simple market features

Targets:

- true latent planted alpha, or
- future return or excess return, depending on intended deployment

Metrics:

- correlation with true alpha
- MSE
- directional accuracy
- out-of-sample robustness across held-out seeds

This tests whether predictive structure is detectable before RL touches it.

Gate:

- Detector quality must hold out of sample and improve or stabilize HRL relative to raw inputs.

## Stage 3: Negative-Control Regime Test

Now test the unsupervised regime layer in a world with no real regimes.

Use the same `GBM + AR + noise` world with constant structure. Run the regime model anyway.

Desired result:

- It should not hallucinate meaningful regimes.
- If it invents persistent states or flips constantly based on noise, that is a warning.

This protects the downstream pipeline from a regime layer that manufactures structure.

Gate:

- No-regime data should not produce a persuasive fake regime structure.

## Stage 4: True Regime Switching

Now introduce real regimes, but keep the change minimal.

Example:

- regime A: `phi = +0.7`
- regime B: `phi = 0`
- regime C: `phi = -0.7`

Only now does regime inference become meaningful.

Evaluate the regime layer against hidden ground truth using:

- adjusted Rand index
- mutual information
- persistence
- transition detection delay

Then run the full ablation matrix:

- A: `HRL only`
- B: `alpha + HRL`
- C: `regime + HRL`
- D: `alpha + regime + HRL`
- Oracle: `true alpha + true regime + HRL`

This is the key architecture experiment.

For the first ablation, the detector and regime layers should be frozen and pretrained rather than jointly trained with HRL. Joint training can be explored later as a separate experiment.

Gate:

- If `alpha + regime + HRL` cannot beat `HRL only` in a world explicitly designed so alpha and regime should help, the added complexity is not earning its keep.

## Stage 5: Noise And Detection Boundary

Keep the mechanism simple, but sweep the noise upward.

Track where each component fails:

- At what noise level does the alpha detector fail?
- At what noise level does the regime detector fail?
- At what noise level does HRL stop making money?

This gives a difficulty curve rather than a binary pass or fail result.

It also identifies the true bottleneck:

- signal extraction
- state inference
- policy learning

## Stage 6: Harder Alpha, Not Harder Market

Before making market dynamics more realistic, make the alpha harder while keeping the market simple.

One change at a time:

- weaker AR
- shorter half-life
- nonlinear alpha
- sparse alpha
- time-varying alpha strength

This isolates whether the system can still detect and exploit genuine predictive structure when it is less obvious.

## Stage 7: Harder Market, Same Simple Alpha

Now keep alpha simple and make the market more realistic, one axis at a time:

- stochastic volatility
- volatility clustering
- fat tails

The key null to protect is that the bot must not convert volatility structure into directional alpha when no directional alpha was planted.

## Stage 8: Realistic Frictions

Before real data, add trading realism:

- commission
- spread
- slippage
- latency
- funding
- market impact

The question now becomes gross-to-net survival, not just gross profitability.

## Stage 9: Adversarial Synthetic Worlds

Now intentionally try to break the system.

Examples:

- in-sample persistence that vanishes out of sample
- alpha decay
- sign reversal
- misleading distractor features
- regime changes before they become obvious

This stage applies overfitting pressure intentionally rather than discovering it by accident.

## Stage 10: Controlled Distribution Shift

Train on one set of conditions and test on unseen ones.

Examples:

- train on A, test on B
- train on A + B, test on C
- train on moderate noise, test on higher noise
- train on one alpha half-life, test on a shorter half-life

The key metric is not simply whether return remains positive. The key question is how gracefully performance degrades under controlled shift.

## Stage 11: Real Data With Ground-Truth Help

Before moving to pure historical data, preserve some ground truth.

### 11a. Real Historical Paths Plus Synthetic Planted Alpha

Inject known alpha into real price paths.

This gives:

- realistic market noise
- known signal truth

This is one of the most valuable intermediate stages because realism increases without losing identifiability.

### 11b. Pure Historical Data

Only here should the project fully give up ground truth.

At this point, the validation harness must already be trusted.

## Null And Control Principle

Every major stage should include an explicit null or control case where the system is expected not to find alpha.

Examples:

- GBM with no planted alpha
- constant world with no true regimes
- harder volatility without directional edge
- realistic market structure with no injected predictive mechanism

The system must demonstrate both sensitivity and specificity:

- sensitivity: detect genuine planted signal
- specificity: do not manufacture signal where none exists

## Validation Rules

Validation is not a final ritual. It is a gate at every major rung.

Use:

- walk-forward evaluation
- purged or embargoed splitting where appropriate
- CPCV-style logic only if it is adapted correctly to sequential RL data and preserves temporal causality; otherwise prefer strict walk-forward with purge or embargo
- repeated seeds under identical conditions
- identical train and test paths across architecture comparisons

Define success before seeing results:

- minimum out-of-sample return
- acceptable in-sample to out-of-sample decay
- stability across folds
- stability across seeds
- max tolerated drawdown behavior

Because the project goal is a healthy rising equity curve with shallow and brief drawdowns, prioritize UPI or Martin ratio heavily, not only Sharpe or Calmar.

## Recommended Immediate Next Experiment

Reduce everything to the smallest high-value step:

1. Rerun the current `GBM + AR` setup through the intended walk-forward or CPCV-style harness.
2. Build the smallest architecture ablation on toy data:
   - `HRL only`
   - `predicted alpha + HRL`
   - `true alpha + HRL`
3. Add actual regime switching and expand to:
   - `HRL only`
   - `alpha + HRL`
   - `regime + HRL`
   - `alpha + regime + HRL`
   - `true alpha + true regime + HRL`
4. Compare all variants on the same seeds, same paths, same training budget, and same evaluation splits.

This gives:

- validator calibration
- representation-layer calibration
- architecture ablation
- a clean basis for every harder step that follows

## One Strict Rule

Never increase more than one fundamental difficulty dimension at once.

Do not jump directly from:

- simple AR in GBM

to:

- harder alpha
- harder volatility
- regimes
- frictions
- cross-asset structure
- distribution shift

all at the same time.

Instead:

1. Prove one piece.
2. Add one complication.
3. Measure what broke.
4. Continue only when the failure is understood.

That is the shortest path to a strategy that can survive real out-of-sample scrutiny.

--

Reporting: https://github.com/nsursock/dirty-fin-reports
Mkt Data: https://github.com/nsursock/dirty-mkt-data
Machine Learning: https://github.com/nsursock/dirty-mlx-ml

We must develop those libraries when appropriate during the project.