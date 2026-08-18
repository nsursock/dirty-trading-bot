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
# (auto-loads the config bound to the checkpoint's manifest.json;
#  refuses a config mismatch unless --force)
venv/bin/python scripts/main.py test

# hyperparameter search (multi-tier + pruning, validation bundle only)
venv/bin/python scripts/optim.py --n-trials 20 --val-seeds 2

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
│   ├── manifest.json        # config hash, dims, seed, git SHAs (every save)
│   └── *.safetensors        # policy checkpoints
└── testing/
    ├── trades.csv           # trade ledger, sorted by close time (+ `episode` col)
    ├── breakdown.txt        # tabulate tables by symbol / exit type
    ├── figure1.png          # aggregate equity, returns, drawdown, return dist
    ├── figure2.png          # aggregate leverage, collateral, long/short, exits
    ├── figure1_episode_<n>  # per-episode figure1 (one per eval.episodes)
    └── figure2_episode_<n>  # per-episode figure2 (one per eval.episodes)
```

## Train / validation / test protocol

Evaluation is split into physically disjoint GBM bundles:

- **`eval.episodes` episodes on seed offsets 1–8** form the *locked final test*.
  `main.py test` / `full` run one episode per offset (default 1) and report an
  aggregate `figure1.png` plus a `figure1_episode_<n>.png` per episode. The
  aggregate equity curve is the average of the per-episode curves, so it is the
  single hold-out number that matters. Per-symbol and per-episode overlay
  curves on the aggregate equity plot are on by default and can be dropped via
  `report.overlays: false`. Test rollouts and the episode loop show tqdm
  progress (`test episodes` / `test seed+<n>`).
- **Seed offsets 10–17** are the *validation bundle*. `optim.py` scores every
  Optuna trial on at least two of these seeds (mean ± CI objective) and never
  reads the locked test. The best trial is additionally deflated with a
  Deflated Sharpe Ratio (Bailey & López de Prado, 2014) using the total trial
  count (`logs/optim_*/optim_result.json`).

Final protocol: tune on the validation bundle → retrain the chosen config →
`main.py test` on the locked test. `test` binds the checkpoint to the exact
config it was trained under via `manifest.json`; a config you did not train
with is refused unless you pass `--force`.

Risk metrics (Sharpe / Sortino / CAGR / Calmar) are always computed from the
bar-indexed `net_curve` of `run_test`, annualized by the low-TF bar duration
(`252 * 1440 / bar_minutes`, e.g. 72,576/yr for 5-minute bars — not a
hard-coded 252). The portfolio is a nominal ~$1000 book: each `eval` position
slot (default `max_positions_per_symbol: 1`, configurable) starts at
`initial_balance` and the portfolio equity is the **mean** of the per-symbol
account curves (not the sum — summing misstates the book size). The
per-trade tables in `breakdown.txt` are descriptive only (no `sqrt(n)`
annualization on trade PnL).

## Margin and return modes

`env.margin_mode` selects the per-slot margin accounting (`isolated`, default,
or `cross`):

- **isolated** — opening a position locks `risk_frac * balance` out of cash
  into collateral; equity = balance + collateral + unrealized PnL and a losing
  position is capped by its collateral.
- **cross** — the whole account equity backs the position: collateral is an
  allocation for sizing/leverage reporting only, cash is *not* locked, position
  sizing compounds off total equity, and liquidation keys off account equity
  falling to the maintenance requirement instead of the position's collateral.

With one position slot per account the two modes differ chiefly in ledger
accounting and the liquidation/bankruptcy path; true portfolio-wide cross
margin (many symbols sharing one $1000 account) is a follow-up redesign of the
per-slot bookkeeping.

`returns.basis` selects the return basis (`account`, default, or `collateral`):

- **account** — every return is measured against total account equity
  (balance + collateral + unrealized PnL), so a +$100 trade on a $1000 book is
  a +10% return no matter how much collateral it used.
- **collateral** — returns are measured against the deployed collateral (ROC).
  The same +$100 trade on $100 collateral is a +100% return. Flowing through
  the training reward (per-bar ROC, zero while flat), the breakdown "By return"
  bucket (PnL ÷ trade collateral), the figure1 return panels, and Sharpe /
  Sortino. Dollar facts (`final_equity`, drawdown, total_return) stay on the
  equity curve either way.

Returns scale: expected per-episode moves are small (sub-percent to a few
percent on the $1000 book). This is by construction of the sizing model —
`risk_frac` allocates only 1–5% of balance per position and take-profit/
stop-loss clamp the PnL — and it is currently dominated by fee/funding drag,
since the trained policies trade nearly every bar (thousands of round trips
per episode). Improving it is a sizing/trade-frequency question; switching to
`returns.basis: collateral` reports the same activity at the true leverage
scale.

## Configuration

`configs/smoke.yaml` (sub-10s smoke) and `configs/normal.yaml` (full run) drive
`scripts/config.py`. Key sections:

```yaml
data:    { n_symbols, n_steps, dt_days }
env:     { n_envs_per_symbol, leverage, margin_mode, fee_rate, funding_rate, ... }
reward:  { mode: smoke|normal, drawdown_penalty }
returns: { basis: account|collateral }
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
configs/       # smoke.yaml, normal.yaml, scalp.yaml, day.yaml, swing.yaml
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

## Environment scaling sweep

Joint-loop throughput (rollout + PPO + SAC) and peak Metal memory vs. `n_envs`
on the `normal.yaml` config (8 symbols, SAC `[256,256]`, PPO `[128,128]`, 1M
replay; M3 Air 16 GB). Each point is the **mean over steady-state cycles**
(first 2 warmup/JIT cycles discarded), so these are sustained numbers, not
single-cycle samples:

| n_envs | steps/sec (mean ± std) | peak Metal (GB) | scaling eff. |
| ---: | ---: | ---: | ---: |
| 256 | 259,881 ± 2,151 | 0.75 | — |
| 512 | 494,399 ± 2,095 | 1.14 | 95% |
| 1024 | 913,951 ± 4,579 | 1.45 | 92% |
| 2048 | 1,524,978 ± 5,584 | 1.96 | 83% |
| 4096 | 1,975,863 ± 96,058 | 2.85 | 65% |

- **Throughput climbs to ~1.98M steps/sec at 4096 envs** and is still scaling
  (+30% over 2048), so the plateau (previously ~1.26M) has not been reached
  within this sweep. 4096 remains compute-bound rather than memory-bound.
- **Memory never swaps** — peak Metal memory 2.85 GB at 4096 envs, process RSS
  steady ~0.3 GB. The binding constraint is **compute**, not memory.
- **Bottleneck is the PPO update** (minibatch `n_steps × n_envs`, so it grows
  with env count). `n_epochs` 10→4 cut it ~2.5×, which is why 2048/4096 got
  ~33–56% faster than the earlier sweep; the per-step cost (VecNormalize +
  intrabar TP/SL) slightly slowed the small 256/512 points.
- A **~3–14% within-run FPS decline** is visible over each point's cycles (a
  few tens of seconds) — consistent with the Air's passive thermal throttling.
  The 20-minute `powermetrics` thermal gate remains the outstanding check.
- **Sweet spot has shifted right to ~2048 envs** (throughput still climbing,
  memory still modest); the default 512-env `normal.yaml` sits comfortably in
  the efficient region with headroom.
