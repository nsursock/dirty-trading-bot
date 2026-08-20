# Dirty Trading Bot

> Research-phase hierarchical reinforcement learning trading bot for synthetic
> perpetuals markets on Apple Silicon.
>
> **Status:** Research | Pre-live | Not for production trading

`dirty-trading-bot` is a quantitative trading research project focused on one
question: can a hierarchical RL stack recover weak directional signal under
controlled synthetic market conditions, survive strict out-of-sample
validation, and eventually earn the right to be tested in paper trading?

The current system is built around a pure-MLX execution path on Apple Silicon:
no PyTorch, no CUDA, no `gymnasium`, and no `stable-baselines3`.

## What This Repo Is

This repository is a research harness, not a live execution bot.

It currently provides:

- a vectorized synthetic perpetuals trading environment
- a hierarchical RL agent with a PPO manager and SAC worker
- train / test / full experiment entry points
- configuration-driven experiments
- reporting with plausibility checks and run artifacts
- utilities for sweeps, benchmarking, and data inspection

It does not yet provide:

- live exchange connectivity
- order routing
- production risk controls
- operational monitoring
- compliance or legal review

## Research Objective

The long-term goal is not just profitability on synthetic data. The goal is to
show that the system can detect and exploit real structure without collapsing
under stronger validation or more realistic frictions.

Current working hypotheses:

- weak planted directional structure can be recovered by the HRL stack
- smoother equity curves matter more than one-off spikes, so drawdown-aware
  evaluation is a first-class concern
- validation discipline should eliminate strategies that only work because of
  leakage, overfitting, or optimistic annualization

The project explicitly favors an equity curve that behaves like a smooth,
recovering staircase rather than a jagged lucky spike. In practice that means
Ulcer-based metrics such as UPI matter alongside Sharpe, Sortino, CAGR, and
max drawdown.

## Research Milestones

- [x] Build the core MLX-based environment and HRL trainer
- [x] Establish a reproducible train / validation / test workflow
- [x] Add synthetic AR-signal experiments and reporting
- [ ] Validate the validator on known planted-alpha worlds (Stage 0)
- [ ] Benchmark ablations: raw HRL vs. alpha-aware variants
- [ ] Add regime-aware / representation-learning layers
- [ ] Introduce more realistic frictions and adversarial shifts
- [ ] Graduate to paper trading only after repeatable out-of-sample evidence

See `roadmap/final_plan.md` for the staged research roadmap.
Stage 0 locked gates live in `roadmap/stage0_gates.yaml` /
`roadmap/stage0_protocol.md`.

## Architecture

The design is intentionally layered:

1. **Phase 1: Hierarchical RL.** Implemented today. A PPO manager emits a
   directional goal every `goal_every` steps and a SAC worker trades under that
   goal.
2. **Phase 2: Supervised alpha detector.** Planned. A directional model will
   sit upstream of the RL layer and serve as a baseline and potentially as an
   input signal.
3. **Phase 3: Unsupervised regime or feature learning.** Planned. This layer is
   intended to capture latent state beyond classical indicators.

Conceptually:

```text
Unsupervised features -> Supervised alpha model -> HRL (PPO Manager + SAC Worker)
```

## Trading Environment

The trading simulator in `scripts/env.py` models one-position-per-symbol
perpetual trading with causal fills and configurable risk assumptions.

Highlights:

- isolated or cross-style per-slot accounting
- taker fees, slippage, and holding-fee drag
- liquidation logic and bankruptcy truncation
- optional take-profit and stop-loss intrabar exits
- discrete or continuous action interfaces
- fully vectorized MLX hot path with `mx.compile`

Reward modes:

- `smoke`: per-step log-equity change
- `normal`: log-equity change minus a drawdown penalty, pushing the policy
  toward a smoother equity curve

## Validation Protocol

This repo treats validation discipline as part of the product.

- locked final tests use disjoint seed bundles
- Optuna tuning uses separate validation seeds and does not read the locked test
- checkpoints are bound to their training config through `manifest.json`
- reporting is delegated to `dirty-fin-reports`, which computes and
  plausibility-checks portfolio metrics

The default workflow is:

1. Tune on the validation bundle.
2. Retrain the chosen config.
3. Evaluate once on the locked final test.

This is meant to make accidental config drift and post-hoc cherry-picking much
harder.

## Setup

### Prerequisites

- Apple Silicon Mac
- Python 3.10+
- local virtual environment at `venv/`
- Chrome for Plotly/Kaleido static image export

Core dependencies:

- `mlx`, `mlx-metal`
- [`dirty-mkt-data`](https://github.com/nsursock/dirty-mkt-data)
- [`dirty-mlx-ml`](https://github.com/nsursock/dirty-mlx-ml)
- [`dirty-fin-reports`](https://github.com/nsursock/dirty-fin-reports)
- `optuna`, `plotly`, `kaleido`, `pandas`, `pyyaml`, `tabulate`, `tqdm`

### Installation

```bash
python3 -m venv venv
venv/bin/pip install \
  "git+https://github.com/nsursock/dirty-mkt-data.git" \
  "git+https://github.com/nsursock/dirty-mlx-ml.git@v0.0.5" \
  "git+https://github.com/nsursock/dirty-fin-reports.git@0.0.4" \
  mlx tqdm pyyaml optuna plotly kaleido pandas tabulate
venv/bin/plotly_get_chrome
```

### Data

This repo currently uses synthetic GBM-based market generation with optional
planted AR structure instead of live exchange data. That is deliberate: the
research phase keeps ground truth under control for as long as possible.

## Running Experiments

Use `venv/bin/python` for all Python entry points.

```bash
# Fast smoke run: train + test + report
venv/bin/python scripts/main.py full --config configs/smoke.yaml

# Full training
venv/bin/python scripts/main.py train --config configs/normal.yaml

# Test the latest checkpoint using its bound config
venv/bin/python scripts/main.py test

# Hyperparameter search on validation seeds
venv/bin/python scripts/optim.py --n-trials 20 --val-seeds 2

# Benchmark harness
venv/bin/python utils/bench.py correctness|env|joint|sweep

# Inspect generated synthetic OHLCV
venv/bin/python utils/inspect_data.py --steps 100

# AR signal-strength sweep
venv/bin/python utils/sweep_ar.py --config configs/exp_ar.yaml

# Stage 0: walk-forward GBM+AR against locked gates
caffeinate -dims venv/bin/python scripts/stage0.py \
  --config configs/stage0_smoke.yaml \
  --gates roadmap/stage0_gates.yaml
```

Each run writes a timestamped folder under `logs/` containing training metrics,
checkpoints, reports, figures, and trade ledgers.

Typical artifacts:

```text
logs/<timestamp>-<pid>/
├── run.log
├── report.json
├── training/
│   ├── manager_ppo.csv
│   ├── worker_sac.csv
│   ├── manager_diag.png
│   ├── worker_diag.png
│   ├── manifest.json
│   └── *.safetensors
└── testing/
    ├── trades.csv
    ├── breakdown.txt
    ├── bot-performance-<verdict>.png
    └── trade-anatomy-<verdict>.png
```

## Environment Sweep

I ran a local `n_envs` sweep on a **MacBook Air M3 with 16 GB RAM** with:

```bash
PYTHONPATH=scripts caffeinate -dims venv/bin/python \
  utils/bench.py sweep_joint \
  --config configs/normal.yaml \
  --seconds 4 \
  --cycles 1 \
  --max-per-sym 512
```

Because `configs/normal.yaml` uses 8 symbols, total `n_envs` below means
`8 x n_envs_per_symbol`.

| Total `n_envs` | Env FPS | Joint-train FPS | Joint peak memory (MB) | Swap used (MB) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 16,791 | 4,789 | 236 | 2,603 |
| 16 | 39,913 | 11,217 | 250 | 2,603 |
| 32 | 80,035 | 22,084 | 286 | 2,603 |
| 64 | 158,800 | 41,769 | 351 | 2,603 |
| 128 | 315,763 | 86,676 | 477 | 2,603 |
| 256 | 622,507 | 169,394 | 731 | 2,603 |
| 512 | 1,229,436 | 337,757 | 1,119 | 2,603 |
| 1,024 | 2,358,209 | 616,239 | 1,410 | 2,603 |
| 2,048 | 4,450,892 | 1,102,306 | 1,884 | 2,603 |
| 4,096 | 8,376,014 | 1,586,616 | 2,708 | 3,249 |

Takeaways:

- collection throughput scaled nearly linearly through the full sweep, reaching
  about **8.38M env-steps/sec** at `4096` total envs
- end-to-end joint throughput reached about **1.59M env-steps/sec** at `4096`
  total envs
- joint peak memory rose from **236 MB** at `8` envs to **2.71 GB** at `4096`
  envs
- swap stayed flat through `2048` envs and only increased materially at
  `4096`, so that top point is usable but closer to the machine's comfort edge

For this repo, FPS and memory both look strongest in the `2048` to `4096` total
env range, while `512` to `2048` looks like the safer day-to-day operating
region if you want substantial throughput with more memory headroom.

## Key Configuration Knobs

The main experiment surface lives in `configs/*.yaml`.

Important sections:

```yaml
data:    { n_symbols, n_steps, dt_days }
env:     { n_envs_per_symbol, leverage, margin_mode, open_fee_rate, close_fee_rate }
reward:  { mode: smoke|normal, drawdown_penalty }
returns: { basis: account|collateral }
train:   { total_timesteps, log_interval }
hrl:     { goal_every, goal_dim }
manager: { n_steps, batch_size, net_arch }
worker:  { buffer_size, learning_starts, net_arch }
```

The baseline full config is `configs/normal.yaml`. The fastest reproducibility
check is `configs/smoke.yaml`.

## Repository Structure

```text
configs/               experiment configurations
roadmap/               research plans and staged roadmap
scripts/
  agents.py            PPO manager + SAC worker training loop
  config.py            YAML loader
  data.py              synthetic market generation
  env.py               vectorized perpetuals simulator
  main.py              train / test / full entry point
  optim.py             Optuna search on validation bundles
  report.py            evaluation and reporting glue
tests/                 focused regression tests
  conftest.py          shared pytest fixtures and setup
  test_ar_gbm.py       AR-vs-GBM data generation checks
  test_exits.py        stop-loss / take-profit behavior
  test_margin.py       margin and liquidation coverage
  test_min_hold.py     minimum holding-period rules
  test_p0.py           core smoke/regression coverage
  test_portfolio.py    portfolio-accounting checks
  test_report_enrich.py report enrichment checks
utils/
  bench.py             performance and correctness harness
  calibrate_ar_noise.py AR-noise calibration helper
  compare_gbm_ar.py    GBM-vs-AR comparison utility
  inspect_data.py      synthetic data visualization
  sweep_ar.py          AR signal-strength sweep runner
```

## Key Files

- `scripts/main.py`: main CLI for training, testing, and full experiment runs
- `scripts/env.py`: vectorized market simulator and reward mechanics
- `scripts/agents.py`: hierarchical RL training logic
- `scripts/optim.py`: validation-only hyperparameter search
- `utils/sweep_ar.py`: controlled signal-strength experiments
- `roadmap/final_plan.md`: staged research plan and validation gates

## Go-Live Criteria

This project should not move toward live capital until all of the following are
true:

- [ ] out-of-sample results remain positive across multiple held-out seeds
- [ ] performance survives stronger frictions and distribution shift
- [ ] drawdowns remain shallow and recover quickly
- [ ] no evidence of leakage, config mismatch, or validation contamination
- [ ] paper trading demonstrates stable behavior
- [ ] operational safeguards exist for execution, monitoring, and failure handling
- [ ] legal, tax, and compliance constraints are reviewed for the target venue

## Contributing

Contributions are welcome, especially around:

- experiment design
- validation methodology
- reward shaping and risk metrics
- synthetic world design
- ablation testing
- performance optimization on MLX

When contributing:

1. Create a feature branch.
2. Keep experiments reproducible and config-driven.
3. Add or update focused tests in `tests/` when behavior changes.
4. Include the rationale for the change and the expected research impact.
5. If applicable, attach before/after backtest or validation evidence.

## Disclaimer

This is a research project. It is not financial advice, and it is not ready for
live trading.

- Synthetic-data success does not imply live-market edge.
- Never commit exchange credentials or secrets.
- Do not connect this code to real capital without a separate productionization,
  controls, and review phase.

## Roadmap

| Phase | Goal |
| --- | --- |
| Research now | Validate known synthetic alpha under strict out-of-sample gates |
| Research next | Add alpha-detector and regime-layer ablations |
| Pre-alpha | Introduce realistic frictions, harder markets, and adversarial worlds |
| Alpha | Paper trading only |
| Beta | Small-scale live pilot with hard risk limits |
| Production | Only if edge, controls, and operations all hold up |

## License

No open-source license is currently granted for this repository. Until a
license file is added, this project should be treated as **all rights
reserved**.

If broader sharing or outside contributions become a goal later, add an
explicit license at that time.
