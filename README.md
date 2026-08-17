# dirty-trading-bot

A high-performance, multi-crypto perpetuals trading bot with backtesting and
[Hierarchical Reinforcement Learning](https://en.wikipedia.org/wiki/Hierarchical_reinforcement_learning),
built entirely on **Apple MLX** (unified memory, Metal GPU) for Apple Silicon.

No PyTorch, no CUDA, no `gymnasium`, no `stable-baselines3` — the RL layer is a
pure-MLX port with an SB3-shaped API (`learn` / `predict`), and the hot path is
a single `mx.compile`-fused kernel running 512–1024 vectorized environments.

```
Unsupervised features  ──>  Supervised baseline  ──>  HRL (PPO Manager + SAC Worker)
      (Phase 3)                  (Phase 2)                  (Phase 1 — implemented)
```

## Architecture

Three layers, built top-down:

1. **Reinforcement learning layer (Phase 1 — implemented).** A two-tier joint
   trainer (`scripts/agents.py`): a discrete **PPO Manager** sets a directional
   goal every `goal_every` worker steps; a continuous **SAC Worker** executes
   with that goal appended to its observation. Both update inside one loop over
   a shared timeline.
2. **Supervised baseline (Phase 2 — planned).** A calibrated directional model
   to sit upstream of the RL decision layer.
3. **Unsupervised feature engineering (Phase 3 — planned).** Learned
   representations / regime latent states, beyond the classical TA stack.

### Trading environment (`scripts/env.py`)

A vectorized, per-symbol perpetuals simulator. Each environment instance trades
one symbol with at most one open position (`flat / long / short`) on isolated
margin, with:

- taker fees, entry slippage, funding accrual
- maintenance-margin liquidation + liquidation penalty
- bankruptcy truncation at `balance < min_collateral` (10 USDC)
- causal fills — act on bar `t`, fill at bar `t+1`
- discrete (`Flat/Long/Short`) or continuous (`[-1, 1]` → discretized) action
  spaces; optional goal conditioning

The `step` is a single `mx.compile`-fused kernel with no Python loops over the
batch axis, and it threads state/time/RNG exactly the way
[`dirty-mlx-ml`](https://github.com/nsursock/dirty-mlx-ml) expects, so its PPO
and SAC consume it directly.

### Reward

Configurable via `reward.mode`:

- `smoke` — per-step log-equity change.
- `normal` — log-equity change minus a quadratic drawdown penalty (the *Ulcer*
  term), steering the policy toward a smooth "mountain-ridge" equity curve.

## Dependencies

Python 3.10+ on Apple Silicon.

| Package | Why |
| --- | --- |
| `mlx`, `mlx-metal` | hot-path tensors + Metal GPU |
| [`dirty-mkt-data`](https://github.com/nsursock/dirty-mkt-data) | synthetic GBM OHLCV generator (MLX only) |
| [`dirty-mlx-ml`](https://github.com/nsursock/dirty-mlx-ml) | SB3-shaped PPO / SAC ports |
| `tqdm`, `pyyaml` | progress bars, config |
| `optuna` | hyperparameter search with pruning |
| `plotly`, `kaleido`, `tabulate` | reporting figures + tables |

## Install

```bash
python3 -m venv venv
venv/bin/pip install \
  "git+https://github.com/nsursock/dirty-mkt-data.git" \
  "git+https://github.com/nsursock/dirty-mlx-ml.git" \
  mlx tqdm pyyaml optuna plotly kaleido tabulate
# kaleido PNGs need Chrome-for-Testing:
venv/bin/plotly_get_chrome
```

## Usage

```bash
# smoke: sub-10s train + test
venv/bin/python scripts/main.py full --config configs/smoke.yaml

# full training (512 parallel envs)
venv/bin/python scripts/main.py train --config configs/normal.yaml

# test the latest checkpoint -> trade ledger + figures
venv/bin/python scripts/main.py test

# hyperparameter search (multi-tier + pruning)
venv/bin/python scripts/optim.py --n-trials 20

# benchmark harness
venv/bin/python scripts/bench.py correctness|env|joint|sweep

# render synthetic OHLCV candlesticks
venv/bin/python utils/inspect_data.py --steps 100
```

Every run writes a timestamped folder `logs/<timestamp>/`:

```
logs/<timestamp>/
├── run.log                  # INFO + DEBUG (file only; stdout stays tqdm)
├── training/
│   ├── manager_ppo.csv      # SB3-style rollout/train/time stats
│   ├── worker_sac.csv
│   ├── manager_diag.png     # 4x3 diagnostic per agent
│   ├── worker_diag.png
│   └── *.safetensors        # policy checkpoints
└── testing/
    ├── trades.csv           # trade ledger (per reporting.md)
    ├── breakdown.txt        # tabulate tables by symbol / exit type
    ├── figure1.png          # equity, returns, drawdown, return dist
    └── figure2.png          # leverage, collateral, long/short, exits
```

## Configuration

`configs/smoke.yaml` (sub-10s smoke) and `configs/normal.yaml` (full run) drive
`scripts/config.py`. Key sections:

```yaml
data:    { n_symbols, n_steps, dt_days }
env:     { n_envs_per_symbol, leverage, fee_rate, funding_rate, ... }
reward:  { mode: smoke|normal, drawdown_penalty }
train:   { total_timesteps, log_interval }
hrl:     { goal_every, goal_dim }
manager: { n_steps, batch_size, net_arch, ... }   # PPO
worker:  { buffer_size, learning_starts, net_arch, ... }  # SAC
```

`normal.yaml` runs `8 symbols × 64 worlds = 512 parallel environments`.

## Project layout

```
scripts/
  config.py    # YAML loader
  data.py      # synthetic GBM OHLCV + pure-MLX TA feature tensor
  env.py       # vectorized perpetuals env (fused step)
  agents.py    # PPO Manager + SAC Worker joint trainer
  main.py      # train / test / full
  report.py    # ledger + figures + ML-health reporting
  optim.py     # Optuna multi-tier HP search
  bench.py     # milestone benchmark harness
utils/
  inspect_data.py  # OHLCV -> PNG
configs/       # smoke.yaml, normal.yaml
sota/          # literature notes (backtesting, microstructure, ...)
```

## Constraints

- **No `gymnasium` / `stable-baselines3`.** The RL layer is a pure-MLX port
  with SB3 parity.
- **Hot path is pure MLX** — no NumPy/Pandas/Python loops in the env kernel.
- **Logging protocol** — `run.log` (INFO+DEBUG) at the run-folder root; stdout
  restricted to `tqdm` progress bars.

## Final scorecard

Measured on a MacBook Air **M3 16 GB**, smoke network sizes
(PPO `[64,64]`, SAC `[128,128]`), synthetic GBM data. See
`milestones/consensus.md` for the full synthesis and the adviser targets.

| Priority | Milestone | Result |
| --- | --- | --- |
| P0 | Correctness baseline | 20 cycles, 0 NaN, mean_win stable, FPS 713→786 (+10% warmup, **no decay**) |
| P0 | Memory/FPS instrumentation | `run.log` logs per-cycle FPS + active/peak memory |
| P1 | Rollout @ 64/128 envs | 10.1k / 20.0k env-steps/sec, **99% scaling efficiency** |
| P1 | Joint loop @ 128 envs | **6,196 steps/sec** (target 1.5–4k), 81 MB |
| P1 | SAC replay-buffer budget | 1M buffer = **208 MB** (obs_dim 24), well under 9 GB |
| P2 | 256-env sweet spot | **12,414 steps/sec** (2× 128), 145 MB |
| P2 | Saturation scan | env: 2.7k→169k steps/sec (16→1024); joint: 6.2k→12.4k→**24.4k** @ 128/256/512 |
| P2 | Thermal stability | **-1.4% FPS drift** over 2.9M steps, 306 MB stable |
| P3 | Optimization pass | `eval_every` 8→32: +37% env / +6% joint → **26k steps/sec @ 512** |
| P3 | Long-run stability | memory plateaus (274–306 MB), no growth |

**Net:** the system is past the consensus targets — ~26k joint steps/sec at
512 envs vs. an 8k+ "platinum" goal. Remaining optional work: compile the
worker rollout into a single `mx.compile` graph, and the full 20-minute
`powermetrics` thermal gate.
