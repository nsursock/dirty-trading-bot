"""Financial + ML health reporting and plotting engine.

``run_test`` drives a trained joint policy deterministically on a fresh
(synthetic) test bundle, reconstructs the trade ledger from the env state,
and emits: the ledger CSV, a ``tabulate`` breakdown, Figure 1 (equity /
returns / drawdown / return distribution) and Figure 2 (leverage / collateral
/ direction / exit types). ``ml_health`` reads the SB3-style progress CSVs
and renders a 4x3 diagnostic figure per agent.
"""

from __future__ import annotations

import csv
import logging
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate
from tqdm import tqdm

from data import SYMBOLS, TRADING_DAYS, build_high_view, generate, mgr_obs
from env import TradingEnv

log = logging.getLogger("trading")

# The locked final-test seed offsets. Optuna / hyperparameter search must
# NEVER read these; ``validate`` uses the validation bundle instead.
# The first ``len(TEST_SEED_OFFSETS)`` ``main.py test`` / ``full`` episodes
# replay exactly these draws; ``eval.episodes`` beyond that count extends the
# sequence deterministically (offsets 9..N) via ``_episode_offsets``.
TEST_SEED_OFFSETS = (1, 2, 3, 4, 5, 6, 7, 8)
# Held-out validation bundle for hyperparameter search (mean +/- CI).
VALID_SEED_OFFSETS = (10, 11, 12, 13, 14, 15, 16, 17)


def _np(x) -> np.ndarray:
    # MLX arrays expose __dlpack__, so np.asarray() maps them directly instead
    # of forcing a Python-level .tolist() round-trip (the old path cost ~7us
    # per call and dominated the test rollout).
    return np.asarray(x)


def _one_hot(a, n):
    import mlx.core as mx

    return mx.equal(mx.arange(n), a[:, None]).astype(mx.float32)


def _test_env(cfg, seed_offset=1):
    import mlx.core as mx

    d = dict(cfg.get("data", {}))
    e = dict(cfg.get("env", {}))
    r = dict(cfg.get("reward", {}))
    h = dict(cfg.get("hrl", {}))
    ev = dict(cfg.get("eval", {}))
    tf = d.get("timeframes", {})
    # Number of independent position-slots per symbol during evaluation
    # (one slot can hold at most one position; default 1 slot => max one
    # open position per symbol).
    e["n_envs_per_symbol"] = max(1, int(ev.get("max_positions_per_symbol", 1)))
    e["reward_mode"] = r.get("mode", "smoke")
    e["drawdown_penalty"] = r.get("drawdown_penalty", 1.0)
    e["return_basis"] = dict(cfg.get("returns", {})).get("basis", "account")
    e["goal_dim"] = h.get("goal_dim", 3)
    e["action_space"] = "continuous"
    ak = ev.get("adaptive_knob")
    if ak:
        e["adaptive_knob"] = ak
    symbols = dict(list(SYMBOLS.items())[: d.get("n_symbols", 4)])
    seed = cfg.get("seed", 42) + seed_offset
    bundle = generate(
        symbols=symbols,
        n_steps=d.get("n_steps", 400),
        seed=seed,
        low_tf=tf.get("low", 5),
        high_tf=tf.get("high", 240),
        regime=d.get("regime", "bull"),
    )
    log.info("test seed+%d: GBM bundle generated (n_steps=%d, n_resample=%d, low=%s high=%s)",
             seed_offset, bundle.features.shape[1], bundle.n_resample,
             tf.get("low", 5), tf.get("high", 240))
    env = TradingEnv(bundle.features, bundle.ohlcv.closes,
                     highs=bundle.ohlcv.highs, lows=bundle.ohlcv.lows, seed=seed, **e)
    return bundle, env, list(bundle.symbols), e


def _finish_trade(tr, exit_price, exit_step, exit_type, fee, realized):
    tr.update(
        {
            "exit_price": exit_price,
            "closed_at": exit_step,
            "exit_type": exit_type,
            "fee": round(tr.get("fee", 0.0) + fee, 6),
            "realized_pnl": realized,
        }
    )
    return tr


def _mgr_obs_test(high_feats, sym_off_hi, worker_obs, low_steps, T_hi, F, n_resample):
    import mlx.core as mx

    acct = worker_obs[:, F : F + 6]
    return mgr_obs(high_feats, acct, low_steps, sym_off_hi, T_hi, n_resample)


def run_test(cfg, manager, worker, norm_state=None, seed_offset=1, pbar=None,
             deterministic=True) -> dict:
    """Joint rollout -> ledger + per-step series.

    ``deterministic=True`` drives both tiers greedily (identical, reproducible
    test runs); ``deterministic=False`` samples manager goals and worker
    actions from the trained policies, so each run exercises the full action
    distribution at the cost of run-to-run variance.

    ``seed_offset`` selects the GBM bundle draw: the locked final test uses
    ``TEST_SEED_OFFSETS``; validation uses ``VALID_SEED_OFFSETS``. When a
    ``pbar`` is passed it is shared across episodes (one progress bar for the
    whole test phase) and this call only advances it; otherwise a per-episode
    bar is created.
    """
    import mlx.core as mx
    from dirty_mlx_ml.reinforcement import VecNormalize

    t_env = time.monotonic()
    bundle, env, symbols, e = _test_env(cfg, seed_offset=seed_offset)
    log.debug("test seed+%d: bundle+env built in %.2fs (n_steps=%d, S=%d, T_hi=%d, n_resample=%d)",
              seed_offset, time.monotonic() - t_env, bundle.features.shape[1], env.n_symbols,
              bundle.high_features.shape[1], bundle.n_resample)
    if norm_state is not None:
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        for k, v in norm_state.items():
            if k != "returns":
                env.norm_state[k] = mx.array(v)
    log.info("test: env num_envs=%d n_symbols=%d T=%d", env.num_envs, env.n_symbols, env.T)
    goal_every = cfg.get("hrl", {}).get("goal_every", 4)
    goal_dim = cfg.get("hrl", {}).get("goal_dim", 3)
    F, init_bal = env.F, e.get("initial_balance", 1000.0)
    n_envs = env.num_envs
    S = env.n_symbols
    n_resample = bundle.n_resample
    T_hi = bundle.high_features.shape[1]
    sym_idx = [s for s in range(S) for _ in range(env.n_envs_per_symbol)]
    sym_off_hi = mx.array([s * T_hi for s in sym_idx], dtype=mx.int32)
    high_feats = build_high_view(bundle.high_features)
    side_thr = e.get("side_threshold", 0.2)
    open_fee_rate = e.get("open_fee_rate", 3e-4)
    close_fee_rate = e.get("close_fee_rate", 6e-4)
    liq_fee_rate = e.get("liquidation_fee_rate", 3e-3)
    mark_impact = e.get("liq_mark_impact", 0.005)
    funding = e.get("holding_fee_daily", 1.5e-4) / max(int(e.get("bars_per_day", 288)), 1)
    T = env.T

    worker_obs = env.reset()[0]
    mgr_obs = _mgr_obs_test(high_feats, sym_off_hi, worker_obs, mx.zeros((n_envs,), mx.int32),
                            T_hi, F, n_resample)

    open_trades = [None] * n_envs
    ledger = []
    prev_side = np.zeros(n_envs, dtype=int)

    # Ledger-derived bookkeeping: reported curves (net/gross/roc/per-symbol)
    # are built from the ledger (trades.csv), not from the live env poll, so
    # txt/png always reconcile with the history CSV. Realized equity steps
    # only at trade closes; open positions contribute nothing until closed.
    book = np.full(n_envs, init_bal, dtype=float)
    fees_paid = np.zeros(n_envs, dtype=float)
    fund_paid = np.zeros(n_envs, dtype=float)

    net_curve, gross_curve, step_axis = [], [], []
    lev_all, coll_all, sides_all, exits = [], [], [], []
    trade_id = 0

    # Static lookups pulled to numpy once: closes_flat and sym_off never change,
    # so the per-step price lookup becomes pure numpy indexing instead of an
    # mx.take + mx.array round-trip per bar.
    closes_np = np.asarray(env.closes_flat)
    sym_off_np = np.asarray(env.sym_off)

    # Pure-MLX rollout: buffer each step's tensors without forcing a host sync,
    # then materialize the whole trajectory once. Per-step ``np.asarray`` on a
    # lazy MLX array forces a graph eval + device sync (~200us each); buffering
    # collapses 4 * T syncs into 4 total.
    states_buf, acts_buf, steps_buf, exits_buf = [], [], [], []
    t = 0
    own_pbar = pbar is None
    if own_pbar:
        pbar = tqdm(total=T - 1, desc=f"test seed+{seed_offset}", unit="step",
                    colour="magenta", leave=False)
    else:
        pbar.set_description(f"test seed+{seed_offset}")
    t_roll = time.monotonic()
    _TICK = max(T // 10, 1)
    while t < T - 1:
        if t % _TICK == 0 and t > 0:
            el = time.monotonic() - t_roll
            log.debug("test seed+%d: %d/%d steps (%.0f steps/s, %.1fs elapsed)",
                      seed_offset, t, T - 1, t / max(el, 1e-9), el)
        goal, _ = manager.predict(mgr_obs, deterministic=deterministic)
        env.set_goal(_one_hot(goal, goal_dim))
        for _k in range(goal_every):
            if t >= T - 1:
                break
            act, _ = worker.predict(worker_obs, deterministic=deterministic)
            worker_obs, r, done, info = env.step(act)
            t += 1
            states_buf.append(env._state)
            acts_buf.append(act)
            steps_buf.append(env._steps)
            exits_buf.append(info["exit"])
            pbar.update(1)
            if _k == goal_every - 1:
                low_steps = mx.minimum(env._steps, T - 1) if hasattr(env, "_steps") else t
                mgr_obs = _mgr_obs_test(high_feats, sym_off_hi, worker_obs, low_steps, T_hi, F, n_resample)

    if own_pbar:
        pbar.update(pbar.total - pbar.n)
        pbar.close()
    log.debug("test seed+%d: rollout done in %.2fs (T=%d, %.0f steps/s)",
              seed_offset, time.monotonic() - t_roll, T - 1, (T - 1) / max(time.monotonic() - t_roll, 1e-9))

    # Materialize the buffered rollout in one shot, then reconstruct the ledger
    # in numpy.
    t_ledger = time.monotonic()
    states_np = np.asarray(mx.stack(states_buf))
    acts_np = np.asarray(mx.stack(acts_buf)).reshape(len(acts_buf), -1)
    steps_np = np.asarray(mx.stack(steps_buf))
    exits_np = np.asarray(mx.stack(exits_buf)).astype(int)
    _EXIT_NAMES = {0: "market_close", 1: "take_profit", 2: "stop_loss", 3: "liquidation"}

    for k in range(len(states_buf)):
        st = k + 1
        state_k = states_np[k]
        balance = state_k[:, 0]
        q = state_k[:, 1]
        entry = state_k[:, 2]
        collateral = state_k[:, 3]
        act_np = acts_np[k]
        t_idx = np.minimum(steps_np[k], T - 1)
        price = closes_np[sym_off_np + t_idx]
        if env.margin_mode == "cross":
            # Poll the account equity exactly like the env does.
            equity = balance + q * (price - entry)
        else:
            equity = balance + collateral + q * (price - entry)
        notional = np.abs(q) * price
        side = np.sign(q).astype(int)
        req = np.where(np.abs(act_np) > side_thr, np.sign(act_np), 0).astype(int)
        exit_flags = exits_np[k]

        for i in range(n_envs):
            if prev_side[i] == 0 and side[i] != 0:
                qty = float(notional[i] / (price[i] + 1e-9)) * (1.0 if side[i] > 0 else -1.0)
                open_trades[i] = {
                    "trade_id": trade_id,
                    "symbol": symbols[sym_idx[i]],
                    "side": "long" if side[i] > 0 else "short",
                    "opened_at": st,
                    "entry_price": float(price[i]),
                    "qty": qty,
                    "notional": float(notional[i]),
                    "leverage": float(notional[i] / (collateral[i] + 1e-9)),
                    "collateral": float(collateral[i]),
                    "equity_before": float(equity[i]),
                    "fee": open_fee_rate * notional[i],
                    "funding": 0.0,
                    "exit_type": "open",
                }
                trade_id += 1
            elif prev_side[i] != 0 and side[i] == 0:
                tr = open_trades[i]
                exit_price = float(price[i])
                exit_type = _EXIT_NAMES.get(int(exit_flags[i]),
                                            "market_close" if req[i] == 0 else "liquidation")
                if tr is not None:
                    if exit_type == "liquidation":
                        # Isolated margin: liquidation loses the full
                        # collateral (pnl % capped at -100%) plus the
                        # liquidation fee. Fill happens at the mark price
                        # with the capped adverse impact.
                        mark = exit_price * (1.0 - mark_impact * (1.0 if tr["side"] == "long" else -1.0))
                        realized = -float(tr["collateral"])
                        exit_fee = liq_fee_rate * abs(tr["qty"]) * mark
                        exit_price = mark
                    else:
                        exit_fee = close_fee_rate * abs(tr["qty"]) * exit_price
                        realized = (tr["qty"] * (exit_price - tr["entry_price"])
                                    - tr["fee"] - exit_fee - tr["funding"])
                    ledger.append(_finish_trade(tr, exit_price, st, exit_type, exit_fee, realized))
                    book[i] += realized
                    fees_paid[i] += tr["fee"]
                    fund_paid[i] += tr["funding"]
                open_trades[i] = None
                exits.append(exit_type)
            elif prev_side[i] != 0 and side[i] != prev_side[i]:
                tr = open_trades[i]
                exit_price = float(price[i])
                if tr is not None:
                    realized = (tr["qty"] * (exit_price - tr["entry_price"])
                                - tr["fee"]
                                - close_fee_rate * abs(tr["qty"]) * exit_price
                                - tr["funding"])
                    ledger.append(_finish_trade(tr, exit_price, st, "market_close",
                                                close_fee_rate * abs(tr["qty"]) * exit_price, realized))
                    exits.append("market_close")
                    book[i] += realized
                    fees_paid[i] += tr["fee"]
                    fund_paid[i] += tr["funding"]
                qty = float(notional[i] / (price[i] + 1e-9)) * (1.0 if side[i] > 0 else -1.0)
                open_trades[i] = {
                    "trade_id": trade_id,
                    "symbol": symbols[sym_idx[i]],
                    "side": "long" if side[i] > 0 else "short",
                    "opened_at": st,
                    "entry_price": float(price[i]),
                    "qty": qty,
                    "notional": float(notional[i]),
                    "leverage": float(notional[i] / (collateral[i] + 1e-9)),
                    "collateral": float(collateral[i]),
                    "equity_before": float(equity[i]),
                    "fee": open_fee_rate * notional[i],
                    "funding": 0.0,
                    "exit_type": "open",
                }
                trade_id += 1

            if side[i] != 0:
                lev_all.append(notional[i] / (collateral[i] + 1e-9))
                coll_all.append(collateral[i])
                sides_all.append("long" if side[i] > 0 else "short")
            if open_trades[i] is not None:
                # Signed funding: the env does `balance -= funding * (q*price)`,
                # so longs pay (+cost) and shorts receive (-cost).
                open_trades[i]["funding"] += funding * notional[i] * float(side[i])

        prev_side = side
        step_axis.append(st)
        # Portfolio equity = mean of the per-account realized books (each
        # env starts at the config initial_balance, so the mean curve
        # tracks one $1000 book that steps only when trades close).
        net_curve.append(float(np.mean(book)))
        gross_curve.append(float(np.mean(book + fees_paid + fund_paid)))

    log.debug("test rollout done: steps=%d trades=%d final_equity=%.4f", t, len(ledger), net_curve[-1])
    log.debug("test seed+%d: ledger/reconstruction phase %.2fs this episode", seed_offset,
              time.monotonic() - t_ledger)
    sym_by_env = np.asarray(sym_idx)

    # Ledger-driven per-symbol equity (realized) and collateral-basis ROC.
    # Each symbol's curve is its cumulative realized PnL scaled back to a
    # single account (divide by the number of position slots per symbol).
    #
    # ROC at a bar is the realized PnL of trades *closing* that bar divided by
    # the collateral of those same trades (not all open collateral): a full
    # liquidation then reads as a ~-100% bar return instead of being diluted
    # by the collateral of still-open positions.
    sym_index = {sym: k for k, sym in enumerate(symbols)}
    T_bars = len(step_axis)
    roc_realized = np.zeros(T_bars, dtype=float)
    roc_coll = np.zeros(T_bars, dtype=float)
    sym_real = np.zeros((S, T_bars), dtype=float)
    for tr in ledger:
        # step_axis[j] = j+1 (t is 1-indexed, j is 0-indexed): a trade closing
        # at bar t maps to index t-1, so a close on the final bar (t=T-1) stays
        # in bounds.
        ca = int(tr["closed_at"]) - 1
        realized = float(tr.get("realized_pnl", 0.0) or 0.0)
        coll = float(tr.get("collateral", 0.0) or 0.0)
        roc_realized[ca] += realized
        roc_coll[ca] += coll
        sym_real[sym_index[tr["symbol"]], ca] += realized
    roc_curve = np.where(roc_coll > 1e-6, roc_realized / np.maximum(roc_coll, 1e-9), 0.0)
    n_per = max(int(env.n_envs_per_symbol), 1)
    per_symbol = init_bal + np.cumsum(sym_real, axis=1) / n_per
    return {
        "ledger": ledger,
        "net": np.asarray(net_curve),
        "gross": np.asarray(gross_curve),
        "steps": np.asarray(step_axis),
        "per_symbol": per_symbol,
        "sym_by_env": sym_by_env,
        "roc": np.asarray(roc_curve),
        "roc_realized": roc_realized,
        "roc_coll": roc_coll,
        "leverage": np.asarray(lev_all),
        "collateral": np.asarray(coll_all),
        "sides": sides_all,
        "exits": exits,
        "seed_offset": seed_offset,
    }


def _sort_ledger(ledger):
    """Sort trades by close time; still-open trades (no close) go last."""

    def _closed(t):
        ca = t.get("closed_at")
        try:
            return float("inf") if ca is None else float(ca)
        except (TypeError, ValueError):
            return float("inf")

    return sorted(
        ledger,
        key=lambda t: (_closed(t), str(t.get("symbol", "")), int(t.get("episode", 0) or 0)),
    )


def write_ledger(ledger, path):
    cols = [
        "trade_id", "episode", "symbol", "side", "opened_at", "closed_at", "entry_price",
        "exit_price", "notional", "leverage", "collateral", "fee", "funding", "realized_pnl", "exit_type",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in ledger:
            w.writerow(t)
    return path


def _returns(equity):
    return np.diff(equity) / (equity[:-1] + 1e-12)


def _periods_per_year(cfg) -> float:
    """Trading periods per year derived from the low-TF bar duration.

    Not a hard-coded 252 (daily): for 5-minute bars this is
    ``252 * 1440 / 5 = 72576``.
    """
    from data import _tf_minutes

    tf = (cfg.get("data") or {}).get("timeframes") or {}
    bar_min = _tf_minutes(tf.get("low", "5m"))
    return TRADING_DAYS * 24 * 60 / bar_min


def metrics(net, periods_per_year=252, rets=None, basis="account"):
    """Risk metrics on a single bar-indexed equity curve.

    ``net`` must be the bar ``net_curve`` from ``run_test`` and
    ``periods_per_year`` derived from ``_periods_per_year(cfg)`` so Sharpe /
    Sortino / CAGR use the true bar cadence. ``rets`` optionally overrides
    the per-period return series (e.g. the collateral-basis ROC series) used
    for Sharpe / Sortino; ``final_equity`` / ``total_return`` / drawdown stay
    on the dollar equity curve. ``basis`` is reported verbatim.
    """
    if rets is None or np.asarray(rets).size == 0:
        rets = _returns(net)
    rets = np.asarray(rets)
    if len(rets) < 2 or np.std(rets) == 0:
        return {}
    mean, std = np.mean(rets), np.std(rets)
    downside = rets[rets < 0]
    dstd = np.std(downside) if len(downside) else 0.0
    peak = np.maximum.accumulate(net)
    dd = (peak - net) / peak
    years = len(net) / periods_per_year
    cagr = (net[-1] / net[0]) ** (1.0 / years) - 1.0 if years > 0 and net[0] > 0 else 0.0
    return {
        "sharpe": mean / (std + 1e-12) * np.sqrt(periods_per_year),
        "sortino": mean / (dstd + 1e-12) * np.sqrt(periods_per_year),
        "max_drawdown": float(np.max(dd)),
        "cagr": cagr,
        "final_equity": float(net[-1]),
        "total_return": float(net[-1] / net[0] - 1.0),
        "return_basis": basis,
    }


def validate(cfg, manager, worker, n_seeds=2, norm_state=None, deterministic=True) -> dict:
    """Score a trained model on the held-out *validation* bundle.

    Runs ``run_test`` on ``n_seeds`` validation seed offsets (never the locked
    ``TEST_SEED_OFFSETS``) and returns the mean + spread of the annualized
    Sharpe. This is the only objective hyperparameter search is allowed to use.
    """
    n_seeds = max(1, int(n_seeds))
    ppy = _periods_per_year(cfg)
    nets, sharpes = [], []
    for off in VALID_SEED_OFFSETS[:n_seeds]:
        r = run_test(cfg, manager, worker, norm_state=norm_state, seed_offset=off,
                     deterministic=deterministic)
        m = metrics(r["net"], periods_per_year=ppy)
        nets.append(np.asarray(r["net"]))
        sharpes.append(float(m.get("sharpe", 0.0)))
    sharpes = np.asarray(sharpes)
    mean = float(np.mean(sharpes))
    std = float(np.std(sharpes)) if sharpes.size > 1 else 0.0
    return {
        "sharpe_mean": mean,
        "sharpe_std": std,
        "sharpe_ci": 1.96 * std / math.sqrt(max(sharpes.size, 1)),
        "sharpe_list": sharpes.tolist(),
        "nets": nets,
        "seed_offsets": list(VALID_SEED_OFFSETS[:n_seeds]),
    }


def _skewness(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    s = float(np.std(x, ddof=1))
    if s == 0.0:
        return 0.0
    return float(np.mean((x - np.mean(x)) ** 3) / s ** 3)


def _kurtosis(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 3.0
    s = float(np.std(x, ddof=1))
    if s == 0.0:
        return 3.0
    return float(np.mean((x - np.mean(x)) ** 4) / s ** 4)


def _phi(z) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _probit(p) -> float:
    """Inverse normal CDF (Acklam's rational approximation, no scipy)."""
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def dsr(net, periods_per_year=252, n_trials=1) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Corrects the annualized Sharpe for multiple testing (``n_trials`), and
    non-normal returns via the sample skewness / kurtosis. ``net`` is the
    bar-indexed equity curve (validation bundle for search trials).
    """
    rets = _returns(np.asarray(net))
    if rets.size < 4:
        return {"deflated_sharpe": 0.0, "dsr_probability": 0.0,
                "expected_max_sharpe": 0.0, "sharpe": 0.0}
    N = rets.size
    mean, std = float(np.mean(rets)), float(np.std(rets))
    if std == 0.0:
        return {"deflated_sharpe": 0.0, "dsr_probability": 0.0,
                "expected_max_sharpe": 0.0, "sharpe": 0.0}
    sr = mean / std
    skew = _skewness(rets)
    kurt = _kurtosis(rets)
    var_sr = max((1.0 / N) * (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr), 1e-12)
    eul = 0.5772156649015329
    if n_trials > 1:
        emax = (1.0 - eul) * _probit(1.0 - 1.0 / n_trials) + eul * _probit(
            1.0 - 1.0 / (n_trials * math.e)
        )
        sr_0 = math.sqrt(var_sr) * emax
    else:
        sr_0 = 0.0
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z_dsr = (sr - sr_0) * math.sqrt(N - 1) / denom
    return {
        "deflated_sharpe": (sr - sr_0) * math.sqrt(periods_per_year),
        "dsr_probability": float(_phi(z_dsr)),
        "expected_max_sharpe": sr_0 * math.sqrt(periods_per_year),
        "sharpe": sr * math.sqrt(periods_per_year),
        "skewness": skew,
        "kurtosis": kurt,
        "n_obs": N,
    }


_BD_COLS = ["label", "num trades", "win rate %", "avg win", "avg loss",
            "net profit", "sharpe", "max dd", "risk reward", "sortino", "calmar", "profit factor"]


def _config_lines(cfg) -> list[str]:
    d = cfg.get("data", {})
    e = cfg.get("env", {})
    r = cfg.get("reward", {})
    h = cfg.get("hrl", {})
    w = cfg.get("worker", {})
    ev = cfg.get("eval", {})
    n_sym = int(d.get("n_symbols", 0) or 0)
    # The eval rollout uses max_positions_per_symbol position slots per symbol
    # (see _test_env), not the training n_envs_per_symbol.
    per_sym = max(1, int(ev.get("max_positions_per_symbol", 1)))
    return [
        f"seed: {cfg.get('seed')}",
        f"symbols: {d.get('n_symbols')}  steps: {d.get('n_steps')}  dt_days: {d.get('dt_days')}",
        f"env: {n_sym} symbols x {per_sym} = {n_sym * per_sym} test envs  "
        f"lev {e.get('lev_min')}–{e.get('lev_max')}x  risk {float(e.get('risk_min',0))*100:.0f}–{float(e.get('risk_max',0))*100:.0f}%  "
        f"eq={e.get('initial_balance')}  fees o/c={e.get('open_fee_rate')}/{e.get('close_fee_rate')}  liq={e.get('liquidation_fee_rate')}  hold/day={e.get('holding_fee_daily')}",
        f"reward: {r.get('mode')}  dd_pen={r.get('drawdown_penalty')}  clip={r.get('reward_clip')}  trade_knob={e.get('trade_knob')}",
        f"hrl: goal_every={h.get('goal_every')}  goal_dim={h.get('goal_dim')}  "
        f"manager n_steps={cfg.get('manager',{}).get('n_steps')}  worker net={w.get('net_arch')} lr={w.get('learning_rate')}",
    ]


def _trade_stats(trades, base=1000.0):
    """Descriptive per-trade stats.

    ``max_dd`` is the peak-to-trough drawdown of the trade-PnL cumsum scaled by
    the total account capital in play (``n_accounts * base``), so a subgroup
    spanning N accounts does not read as an N-fold drawdown. Sharpe / Sortino /
    Calmar are per-trade (NOT annualized): mean/std, mean/downside-std, and
    total-return/max-drawdown respectively.
    """
    pnls = np.array([float(t.get("realized_pnl", 0.0) or 0.0) for t in trades], dtype=float)
    n = int(pnls.size)
    if n == 0:
        return dict(num=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0, net=0.0, sharpe=0.0,
                    max_dd=0.0, rr=0.0, sortino=0.0, calmar=0.0, pf=0.0)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    net = float(pnls.sum())
    win_rate = 100.0 * float((pnls > 0).mean())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    accts = {(int(t.get("episode", 0) or 0), str(t.get("symbol", ""))) for t in trades}
    n_accts = max(len(accts), 1)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd_min = float((cum - peak).min())
    max_dd = 100.0 * abs(dd_min) / max(abs(base) * n_accts, 1.0) if abs(dd_min) > 1e-9 else 0.0
    rr = avg_win / abs(avg_loss) if losses.size and abs(avg_loss) > 1e-12 else 0.0
    gw = float(wins.sum()) if wins.size else 0.0
    gl = float(abs(losses.sum())) if losses.size else 0.0
    pf = gw / gl if gl > 1e-12 else (999.0 if gw > 0 else 0.0)
    std = float(pnls.std())
    sharpe = float(pnls.mean()) / std if n > 1 and std > 1e-12 else 0.0
    dstd = float(losses.std()) if losses.size > 1 else 0.0
    sortino = float(pnls.mean()) / dstd if dstd > 1e-12 else 0.0
    total_ret = net / max(abs(base) * n_accts, 1.0)
    calmar = total_ret / (max_dd / 100.0) if max_dd > 1e-9 else 0.0
    return dict(num=n, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss, net=net,
                sharpe=sharpe, max_dd=max_dd, rr=rr, sortino=sortino, calmar=calmar, pf=pf)


def _bd_row(label, st):
    return [label, st["num"], round(st["win_rate"], 2), round(st["avg_win"], 4),
            round(st["avg_loss"], 4), round(st["net"], 4), round(st["sharpe"], 4),
            round(st["max_dd"], 2), round(st["rr"], 4), round(st["sortino"], 4),
            round(st["calmar"], 4), round(st["pf"], 4)]


def _bd_table(title, groups, portfolio):
    rows = [_BD_COLS]
    for label, st in groups:
        rows.append(_bd_row(label, st))
    rows.append(_bd_row("portfolio", portfolio))
    return [title, "", tabulate(rows, headers="firstrow", tablefmt="grid", floatfmt=".4f",
                                colalign=("left",) + ("right",) * (len(_BD_COLS) - 1)), ""]


def breakdown(result, path, cfg=None, rets=None, basis="account"):
    ledger = result["ledger"]
    net = np.asarray(result["net"])
    base = float(net[0]) if net.size else 1000.0
    ppy = _periods_per_year(cfg) if cfg is not None else 252
    m = metrics(net, periods_per_year=ppy, rets=rets, basis=basis)
    port = _trade_stats(ledger, base=base)
    if m:
        port["sharpe"] = m.get("sharpe", 0.0)
        port["sortino"] = m.get("sortino", 0.0)
        port["max_dd"] = 100.0 * m.get("max_drawdown", 0.0)
        port["calmar"] = m.get("cagr", 0.0) / max(abs(m.get("max_drawdown", 0.0)), 1e-9)

    def _bucket(trades, keyfn, order):
        groups = {}
        for t in trades:
            groups.setdefault(keyfn(t), []).append(t)
        return [(lab, _trade_stats(groups[lab], base=base)) for lab in order if lab in groups]

    n = max(len(ledger), 1)
    by_symbol = sorted({t["symbol"] for t in ledger})
    sym_groups = [(s, _trade_stats([t for t in ledger if t["symbol"] == s], base=base)) for s in by_symbol]

    ep_order = sorted({int(t.get("episode", 0) or 0) for t in ledger})
    by_episode = []
    for e in ep_order:
        ep_trades = [t for t in ledger if int(t.get("episode", 0) or 0) == e]
        off = next((t.get("seed_offset") for t in ep_trades if t.get("seed_offset") is not None), "?")
        by_episode.append((f"episode {e} (seed+{off})", _trade_stats(ep_trades, base=base)))
    if len(ep_order) > 1:
        by_episode.append(("all", _trade_stats(ledger, base=base)))

    by_side = _bucket(ledger, lambda t: "bull (long)" if t["side"] == "long" else "bear (short)",
                      ["bull (long)", "bear (short)"])
    by_exit = _bucket(ledger, lambda t: t["exit_type"],
                      ["take_profit", "stop_loss", "market_close", "liquidation"])
    by_outcome = _bucket(ledger, lambda t: ("win" if (t.get("realized_pnl", 0) or 0) > 0 else
                                            "loss" if (t.get("realized_pnl", 0) or 0) < 0 else "breakeven"),
                         ["win", "loss", "breakeven"])

    def _lev(t):
        v = float(t.get("leverage", 0) or 0)
        if v < 5:
            return "cruiser (1-5x)"
        if v < 10:
            return "charger (5-10x)"
        if v < 20:
            return "turbo (10-20x)"
        if v < 50:
            return "warp (20-50x)"
        if v < 100:
            return "hyper (50-100x)"
        return "singularity (100x+)"
    by_lev = _bucket(ledger, _lev, ["cruiser (1-5x)", "charger (5-10x)", "turbo (10-20x)",
                                    "warp (20-50x)", "hyper (50-100x)", "singularity (100x+)"])

    low_tf = ((cfg.get("data") or {}).get("timeframes") or {}).get("low", "5m") if cfg is not None else "5m"
    from data import _tf_minutes
    bar_secs = float(_tf_minutes(low_tf)) * 60.0

    def _hold(t):
        d = int(t.get("closed_at", 0) or 0) - int(t.get("opened_at", 0) or 0)
        secs = d * bar_secs
        if secs <= 0:
            return "unmatched (0s)"
        if secs < 30:
            return "flash (<30s)"
        if secs < 120:
            return "scalp (30s-2m)"
        if secs < 900:
            return "sprint (2-15m)"
        if secs < 3600:
            return "sit (15-60m)"
        return "camp (1h+)"
    by_hold = _bucket(ledger, _hold, ["unmatched (0s)", "flash (<30s)", "scalp (30s-2m)",
                                      "sprint (2-15m)", "sit (15-60m)", "camp (1h+)"])

    ret_basis = (dict(cfg.get("returns", {})) if cfg is not None else {}).get("basis", "account")

    def _roe(t):
        if ret_basis == "collateral":
            base = float(t.get("collateral", 0) or 0)
        else:
            base = float(t.get("equity_before", 0) or 0)
        r = 10_000.0 * (float(t.get("realized_pnl", 0) or 0)) / max(abs(base), 1e-9)
        if r < 0:
            return "dip (<0 bps)"
        if r < 10:
            return "scratch (<10 bps)"
        if r < 100:
            return "single-R (10-100 bps)"
        return "multi-R (>=100 bps)"
    by_roe = _bucket(ledger, _roe, ["dip (<0 bps)", "scratch (<10 bps)",
                                    "single-R (10-100 bps)", "multi-R (>=100 bps)"])

    def _collateral(t):
        v = float(t.get("collateral", 0) or 0)
        if v < 50:
            return "pocket (<$50)"
        if v < 200:
            return "small ($50-$200)"
        if v < 1000:
            return "standard ($200-$1k)"
        return "loaded (>=$1k)"
    by_collateral = _bucket(ledger, _collateral, ["pocket (<$50)", "small ($50-$200)",
                                                  "standard ($200-$1k)", "loaded (>=$1k)"])

    def _notional(t):
        v = float(t.get("notional", 0) or 0)
        if v < 1000:
            return "toy (<$1k)"
        if v < 10_000:
            return "standard ($1k-$10k)"
        if v < 50_000:
            return "size ($10k-$50k)"
        return "whale (>=$50k)"
    by_notional = _bucket(ledger, _notional, ["toy (<$1k)", "standard ($1k-$10k)",
                                              "size ($10k-$50k)", "whale (>=$50k)"])

    def _heat(t):
        eq = float(t.get("equity_before", base) or base)
        r = eq / max(base, 1e-9)
        if r < 0.80:
            return "underwater (<80% eq)"
        if r < 0.95:
            return "bruised (80-95% eq)"
        if r < 1.05:
            return "par (95-105% eq)"
        if r < 1.30:
            return "green (105-130% eq)"
        return "moon (>=130% eq)"
    by_heat = _bucket(ledger, _heat, ["underwater (<80% eq)", "bruised (80-95% eq)", "par (95-105% eq)",
                                      "green (105-130% eq)", "moon (>=130% eq)"])

    def _bite(t):
        coll = max(abs(float(t.get("collateral", 0) or 0)), 1e-9)
        pct = 100.0 * abs(float(t.get("realized_pnl", 0) or 0)) / coll
        if pct < 0.5:
            return "dust (<0.5% margin)"
        if pct < 2:
            return "scratch (0.5-2%)"
        if pct < 8:
            return "nibble (2-8%)"
        if pct < 20:
            return "bite (8-20%)"
        return "feast (>=20%)"
    by_bite = _bucket(ledger, _bite, ["dust (<0.5% margin)", "scratch (0.5-2%)", "nibble (2-8%)",
                                      "bite (8-20%)", "feast (>=20%)"])

    def _liq(t):
        lev = max(abs(float(t.get("leverage", 0) or 0)), 1e-9)
        e = (cfg.get("env") or {}) if cfg is not None else {}
        base = float(e.get("liq_threshold_base", 0.90))
        floor = float(e.get("liq_threshold_floor", 0.67))
        ref = float(e.get("liq_threshold_ref_lev", 2.0))
        hi = float(e.get("liq_threshold_hi_lev", 150.0))
        lo = float(e.get("liq_threshold_lo_lev", 1.0))
        slope_lo = (base - 1.0) / max(ref - lo, 1e-6)
        slope_hi = (floor - base) / max(hi - ref, 1e-6)
        thr = (1.0 + slope_lo * (lev - lo)) if lev <= ref else (base + slope_hi * (lev - ref))
        thr = min(1.0, max(floor, thr))
        # ``thr`` is the perp loss fraction at which liquidation fires, so the
        # required adverse underlying move is ``thr / leverage``.
        dist = 100.0 * thr / lev
        if dist < 2:
            return "knife-edge (<2%)"
        if dist < 5:
            return "tight (2-5%)"
        if dist < 15:
            return "cushion (5-15%)"
        return "fortress (>=15%)"
    by_liq = _bucket(ledger, _liq, ["knife-edge (<2%)", "tight (2-5%)",
                                    "cushion (5-15%)", "fortress (>=15%)"])

    def _fee_drag(t):
        notional = max(abs(float(t.get("notional", 0) or 0)), 1e-9)
        bps = 10_000.0 * float(t.get("fee", 0) or 0) / notional
        if bps <= 0:
            return "free (maker)"
        if bps < 5:
            return "light (<5 bps)"
        return "heavy (>=5 bps)"
    by_fee = _bucket(ledger, _fee_drag, ["free (maker)", "light (<5 bps)", "heavy (>=5 bps)"])

    margin_mode = ((cfg.get("env") or {}).get("margin_mode", "isolated")) if cfg is not None else "isolated"
    by_margin = _bucket(ledger, lambda t: ("pool (cross)" if margin_mode == "cross" else "loner (isolated)"),
                        ["loner (isolated)", "pool (cross)"])

    def _vintage(i):
        frac = i / n
        return "opening act (first 20%)" if frac < 0.2 else "encore (last 20%)" if frac >= 0.8 else "mid-set (20-80%)"
    _vgroups = {}
    for _i, _t in enumerate(ledger):
        _vgroups.setdefault(_vintage(_i), []).append(_t)
    by_vintage = [(lab, _trade_stats(_vgroups[lab], base=base))
                  for lab in ("opening act (first 20%)", "mid-set (20-80%)", "encore (last 20%)")
                  if lab in _vgroups]

    lines: list[str] = ["BREAKDOWN", "=========", ""]
    if cfg is not None:
        lines += _config_lines(cfg)
        lines.append("")
    lines.append("Portfolio risk (Sharpe/Sortino/Calmar) is computed from the bar-indexed "
                 "net curve; subgroup rows report descriptive trade stats only.")
    lines.append(f"portfolio: {port['num']} trades  final_equity={float(net[-1]):.2f}  "
                 f"ret={m.get('total_return', 0):+.2%}  sharpe={port['sharpe']:.3f}  max_dd={port['max_dd']:.2f}%")
    lines.append("")
    for title, groups in [
        ("By symbol", sym_groups),
        ("By episode", by_episode),
        ("By position direction", by_side),
        ("By exit", by_exit),
        ("By outcome", by_outcome),
        ("By leverage", by_lev),
        ("By hold duration", by_hold),
        ("By return" if ret_basis == "collateral" else "By RoE", by_roe),
        ("By collateral", by_collateral),
        ("By notional", by_notional),
        ("By equity heat", by_heat),
        ("By bite size", by_bite),
        ("By liquidation distance", by_liq),
        ("By fee drag", by_fee),
        ("By margin type", by_margin),
        ("By trade vintage", by_vintage),
    ]:
        lines += _bd_table(title, groups, port)

    lines.append("Baselines (after fees/funding/slip)")
    lines.append(f"  policy: {float(net[0]):.2f} -> {float(net[-1]):.2f}  ({m.get('total_return', 0):+.3f})")
    lines.append(f"  flat: 1000.00 -> 1000.00  (+0.000)")
    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(text)
    return text


# --- iso-trading-bot-style figure scaffolding (synthwave / ghibli themes) ---


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


# Local theme palettes (repo-tracked). Keys mirror dirty-mkt-data's Theme
# dataclass so they slot straight into the shared _palette() mapping.
_LOCAL_THEMES = {
    "cyberpunk": dict(
        background="#0b0b16", plot_background="#121222", grid="#1f1f38",
        text="#e0e0e8", up="#00f3ff", down="#ff0055", accent="#00ff66",
    ),
    "nordic_frost": dict(
        background="#f4f7f6", plot_background="#ffffff", grid="#e2e8e6",
        text="#2e4057", up="#048a81", down="#858ae3", accent="#ff8a5b",
    ),
    "tokyo_midnight": dict(
        background="#1a1b26", plot_background="#24283b", grid="#414868",
        text="#c0caf5", up="#7aa2f7", down="#f7768e", accent="#bb9af7",
    ),
    "wes_anderson": dict(
        background="#fbf9f1", plot_background="#f4efe2", grid="#e6decb",
        text="#33261d", up="#66a182", down="#d1495b", accent="#edae49",
    ),
    "brutalist_terminal": dict(
        background="#000000", plot_background="#000000", grid="#1a1a1a",
        text="#00ff00", up="#00ff00", down="#ff3333", accent="#00ffff",
    ),
    "botanical_dark": dict(
        background="#111b15", plot_background="#18261e", grid="#23372c",
        text="#d8e2dc", up="#4e8752", down="#b48a60", accent="#a3c1ad",
    ),
    "solarized_warm": dict(
        background="#fdf6e3", plot_background="#eee8d5", grid="#e0dcc7",
        text="#657b83", up="#2aa198", down="#cb4b16", accent="#268bd2",
    ),
    "sunset_drive": dict(
        background="#15121f", plot_background="#1f1a2e", grid="#2e2542",
        text="#f0e6ff", up="#ff9966", down="#c71585", accent="#ff5e7e",
    ),
}


def _theme_pool() -> list[str]:
    """All usable theme names: local repo themes + dirty-mkt-data package themes."""
    from dirty_mkt_data.viz.themes import THEMES

    return sorted(_LOCAL_THEMES) + sorted(THEMES)


def resolve_theme(theme: str) -> str:
    """Resolve a requested theme to a concrete name; ``random`` picks one at random."""
    if theme == "random":
        return random.choice(_theme_pool())
    return theme


def _palette(theme: str) -> dict:
    """Map a theme onto the iso-trading-bot ``_C`` palette roles
    (up=positive, down=negative, accent=highlight).

    Local repo themes (see ``_LOCAL_THEMES``: cyberpunk, nordic_frost,
    tokyo_midnight, wes_anderson, brutalist_terminal, botanical_dark,
    solarized_warm, sunset_drive) win; anything else falls through to the
    dirty-mkt-data package themes (synthwave / ghibli / valorant).

    ``muted`` (tick labels / subtitles) is derived from the background's
    brightness so it always has readable contrast on light *and* dark themes.
    """
    from dirty_mkt_data.viz.themes import Theme, THEMES

    local = _LOCAL_THEMES.get(theme)
    if local is not None:
        t = Theme(name=theme, **local)
    else:
        t = THEMES[theme]
    up, down, accent = t.up, t.down, t.accent
    r, g, b = (int(t.background[i : i + 2], 16) for i in (1, 3, 5))
    dark_bg = (r + g + b) / 3.0 < 128.0
    muted = "#9AA4B2" if dark_bg else "#5C6570"
    return {
        "bg": t.background,
        "panel": t.plot_background,
        "ink": t.text,
        "muted": muted,
        "grid": t.grid,
        "spine": t.grid,
        "cyan": up,
        "cyan_soft": _rgba(up, 0.22),
        "cyan_glow": _rgba(up, 0.45),
        "magenta": down,
        "magenta_soft": _rgba(down, 0.28),
        "lime": accent,
        "lime_soft": _rgba(accent, 0.20),
        "violet": accent,
        "amber": accent,
        "slate": muted,
        "long": up,
        "short": down,
    }


_FONT = dict(family="JetBrains Mono, Menlo, Monaco, Consolas, monospace")


def _base_layout(c: dict, **extra) -> dict:
    layout = dict(
        paper_bgcolor=c["bg"],
        plot_bgcolor=c["panel"],
        font=dict(family=_FONT["family"], color=c["ink"]),
        title_font=dict(size=18, color=c["cyan"], family=_FONT["family"]),
        margin=dict(l=60, r=32, t=72, b=52),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, color=c["ink"])),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=c["panel"], bordercolor=c["cyan"],
            font=dict(color=c["ink"], size=11, family=_FONT["family"]),
        ),
    )
    layout.update(extra)
    return layout


def _style_axes(fig, rows, cols, c: dict, pct_y=None, pct_x=None):
    pct_y = pct_y or set()
    pct_x = pct_x or set()
    for r in range(1, rows + 1):
        for cc in range(1, cols + 1):
            fig.update_xaxes(
                row=r, col=cc, showgrid=True, gridcolor=c["grid"], gridwidth=1,
                zeroline=False, showline=True, linecolor=c["spine"], linewidth=1, mirror=False,
                tickfont=dict(size=10, color=c["muted"]), title_font=dict(size=11, color=c["muted"]),
                tickformat=".1%" if (r, cc) in pct_x else None, automargin=True,
            )
            fig.update_yaxes(
                row=r, col=cc, showgrid=True, gridcolor=c["grid"], gridwidth=1,
                zeroline=False, showline=True, linecolor=c["spine"], linewidth=1, mirror=False,
                tickfont=dict(size=10, color=c["muted"]), title_font=dict(size=11, color=c["muted"]),
                tickformat=".1%" if (r, cc) in pct_y else None, automargin=True,
            )


def _write_png(fig, path, width=1280, height=920, scale=2):
    fig.write_image(str(path), format="png", width=width, height=height, scale=scale)


def figure1(result, path, theme="synthwave", overlays=True, ret_series=None):
    c = _palette(theme)
    net, gross = np.asarray(result["net"]), np.asarray(result["gross"])
    steps = np.asarray(result["steps"])
    peak = np.maximum.accumulate(net)
    dd = (net - peak) / np.maximum(peak, 1e-8)
    max_dd = float(dd.min())
    net_ret = float(net[-1] / max(net[0], 1e-8) - 1.0)
    gross_ret = float(gross[-1] / max(gross[0], 1e-8) - 1.0)
    fees = float(np.sum(gross - net))
    n_close = len(result["ledger"])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Equity curve", "Trade returns", "Drawdown", "Return distribution"),
        horizontal_spacing=0.09, vertical_spacing=0.14,
    )

    # cost band between gross and net
    fig.add_trace(go.Scatter(x=steps, y=gross, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=net, mode="lines", fill="tonexty",
                             fillcolor=c["magenta_soft"], line=dict(color=c["cyan"], width=0),
                             showlegend=False, hoverinfo="skip"), 1, 1)

    # per-symbol equity (portfolio decomposition, faint)
    per_symbol = result.get("per_symbol")
    if overlays and per_symbol is not None and np.asarray(per_symbol).size:
        for row in np.asarray(per_symbol):
            fig.add_trace(go.Scatter(x=steps, y=row, mode="lines",
                                     line=dict(color=c["violet"], width=1.2), opacity=0.35,
                                     showlegend=False, hoverinfo="skip"), 1, 1)
    # per-episode equity (multi-episode runs, faint)
    ep_list = result.get("episodes")
    if overlays and ep_list:
        for e in ep_list:
            fig.add_trace(go.Scatter(x=np.asarray(e["steps"]), y=np.asarray(e["net"]), mode="lines",
                                     line=dict(color=c["muted"], width=1.2, dash="dot"), opacity=0.5,
                                     showlegend=False, hoverinfo="skip"), 1, 1)

    # glow under net
    fig.add_trace(go.Scatter(x=steps, y=net, mode="lines", line=dict(color=c["cyan"], width=7),
                             opacity=0.2, showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=net, mode="lines", line=dict(color=c["cyan"], width=2.6), name="Net"), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=gross, mode="lines", line=dict(color=c["lime"], width=1.8, dash="dot"), name="Gross"), 1, 1)
    fig.add_hline(y=float(net[0]), line=dict(color=c["muted"], width=1, dash="dash"), row=1, col=1)

    # True per-trade returns: realized_pnl / collateral at the trade's close
    # bar. The aggregated bar ROC is diluted across episodes at aligned bar
    # indices, so a -100% liquidation never surfaces in it; plotting the
    # ledger's own returns preserves the full picture (liq = -1.0 exactly).
    xs = np.array([float(t.get("closed_at", 0.0) or 0.0) for t in result["ledger"]], dtype=float)
    coll = np.array([float(t.get("collateral", 0.0) or 0.0) for t in result["ledger"]], dtype=float)
    ys = np.where(coll > 1e-9, np.array([float(t.get("realized_pnl", 0.0) or 0.0) for t in result["ledger"]])
                  / np.maximum(coll, 1e-9), 0.0)
    liq = np.array([t.get("exit_type") == "liquidation" for t in result["ledger"]])
    trade_rets = ys  # full-resolution per-trade returns for the histogram
    # Rendering is O(trades): with ~100k+ closes (large episodes / stochastic
    # rollouts) a marker per trade becomes the dominant report cost. Downsample
    # to a fixed cap while keeping the true counts in the title.
    _MAX_TRADE_MARKERS = 12_000
    if ys.size > _MAX_TRADE_MARKERS:
        _cap = _MAX_TRADE_MARKERS
        idx = np.linspace(0, ys.size - 1, _cap).round().astype(int)
        idx = np.unique(idx)
        xs, ys, liq, coll = xs[idx], ys[idx], liq[idx], coll[idx]
        sub_rets = ys
    else:
        sub_rets = ys
    win = sub_rets > 0
    loss = sub_rets <= 0
    fig.add_trace(go.Scatter(x=xs[win], y=sub_rets[win], mode="markers",
                             marker=dict(size=7, color=c["magenta"], opacity=0.75, line=dict(width=0)), name="Win"), 1, 2)
    fig.add_trace(go.Scatter(x=xs[loss & ~liq], y=sub_rets[loss & ~liq], mode="markers",
                             marker=dict(size=7, color=c["cyan"], opacity=0.75, line=dict(width=0)), name="Loss"), 1, 2)
    fig.add_trace(go.Scatter(x=xs[liq], y=sub_rets[liq], mode="markers",
                             marker=dict(size=9, symbol="x", color=c["cyan"], opacity=0.9,
                                         line=dict(width=0)), name="Liquidation"), 1, 2)
    fig.add_hline(y=0, line=dict(color=c["spine"], width=1), row=1, col=2)
    fig.add_hline(y=-1.0, line=dict(color="#FF4D6D", width=1, dash="dash"), row=1, col=2)

    fig.add_trace(go.Scatter(x=steps, y=dd, mode="lines", fill="tozeroy", fillcolor=c["magenta_soft"],
                             line=dict(color=c["magenta"], width=2.4), name="Drawdown", showlegend=False), 2, 1)
    fig.add_trace(go.Histogram(x=trade_rets, nbinsx=36, marker=dict(color=c["cyan"], line=dict(color=c["bg"], width=0.6), opacity=0.9),
                               showlegend=False), 2, 2)
    fig.add_vline(x=0, line=dict(color=c["muted"], width=1, dash="dash"), row=2, col=2)
    fig.add_vline(x=float(np.mean(trade_rets)), line=dict(color=c["lime"], width=1.8), row=2, col=2)

    fig.update_layout(**_base_layout(c, title=dict(
        text=(f"Equity & risk<br><sup style='color:{c['muted']}'>"
              f"Net {net_ret:+.1%} · gross {gross_ret:+.1%} · costs {fees:,.0f} · max DD {max_dd:.1%} · {n_close} closes"
              f"{' · markers sampled @ ' + str(_MAX_TRADE_MARKERS) if ys.size > _MAX_TRADE_MARKERS else ''}</sup>"),
        x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        margin=dict(l=60, r=32, t=72, b=88)))
    fig.update_yaxes(title_text="Equity (USDC)", row=1, col=1)
    fig.update_yaxes(title_text="Return", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_xaxes(title_text="Return", row=2, col=2)
    fig.update_annotations(font=dict(size=13, color=c["lime"]))
    _style_axes(fig, 2, 2, c, pct_y={(1, 2)}, pct_x={(2, 2)})
    _write_png(fig, path)
    return path


def figure2(result, path, theme="synthwave"):
    c = _palette(theme)
    ledger = result["ledger"]
    lev = np.asarray([float(t.get("leverage", 0) or 0) for t in ledger])
    coll = np.asarray([float(t.get("notional", 0) or 0) / (float(t.get("leverage", 0) or 1) or 1) for t in ledger])
    sides = [t.get("side") for t in ledger]
    exits = [t.get("exit_type") for t in ledger]
    exit_colors = {
        "tp": c["cyan"], "take_profit": c["cyan"], "sl": c["amber"], "stop_loss": c["amber"],
        "market": c["violet"], "market_close": c["violet"],
        "liquidation": c["magenta"], "limit": c["slate"], "bankrupt": "#FF4D6D", "none": c["muted"],
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Leverage", "Collateral", "Direction", "Exit type"),
        horizontal_spacing=0.09, vertical_spacing=0.14,
    )

    fig.add_trace(go.Histogram(x=lev if lev.size else [0], nbinsx=24,
                               marker=dict(color=c["cyan"], line=dict(color=c["bg"], width=0.6), opacity=0.92),
                               showlegend=False), 1, 1)
    if lev.size:
        fig.add_vline(x=float(np.median(lev)), line=dict(color=c["lime"], width=1.8, dash="dash"), row=1, col=1)
    fig.add_trace(go.Histogram(x=coll if coll.size else [0], nbinsx=24,
                               marker=dict(color=c["violet"], line=dict(color=c["bg"], width=0.6), opacity=0.92),
                               showlegend=False), 1, 2)
    if coll.size:
        fig.add_vline(x=float(np.median(coll)), line=dict(color=c["lime"], width=1.8, dash="dash"), row=1, col=2)

    cnt = Counter(sides)
    fig.add_trace(go.Bar(x=["Long", "Short"], y=[cnt.get("long", 0), cnt.get("short", 0)],
        marker=dict(color=[c["long"], c["short"]], line=dict(width=0), opacity=0.95),
        text=[cnt.get("long", 0), cnt.get("short", 0)], textposition="outside",
        textfont=dict(size=12, color=c["ink"]), showlegend=False), 2, 1)

    ecnt = Counter(exits)
    order = [k for k in ("take_profit", "stop_loss", "market_close", "liquidation", "tp", "sl", "market", "limit", "bankrupt") if k in ecnt]
    order += [k for k in ecnt if k not in order]
    if not order:
        order, vals = ["none"], [0]
    else:
        vals = [ecnt[k] for k in order]
    colors = [exit_colors.get(k, c["muted"]) for k in order]
    fig.add_trace(go.Bar(x=order, y=vals, marker=dict(color=colors, line=dict(width=0), opacity=0.95),
        text=vals, textposition="outside", textfont=dict(size=12, color=c["ink"]), showlegend=False), 2, 2)

    fig.update_layout(**_base_layout(c, title=dict(text="Trade anatomy", x=0.01, xanchor="left"),
        showlegend=False, bargap=0.28))
    for r in (1, 2):
        fig.update_yaxes(title_text="Count", row=r, col=1)
        fig.update_yaxes(title_text="Count", row=r, col=2)
    fig.update_xaxes(title_text="Leverage (x)", row=1, col=1)
    fig.update_xaxes(title_text="Initial margin (USDC)", row=1, col=2)
    fig.update_xaxes(title_text="Side", row=2, col=1)
    fig.update_xaxes(title_text="Exit type", row=2, col=2)
    fig.update_annotations(font=dict(size=13, color=c["lime"]))
    _style_axes(fig, 2, 2, c)
    _write_png(fig, path)
    return path


_SAC_ALIASES = {
    "train/loss/policy": ("train/actor_loss",),
    "train/loss/critic": ("train/critic_loss",),
    "train/loss/alpha": ("train/ent_coef_loss",),
    "train/policy/alpha": ("train/ent_coef",),
}
PPO_KEYS = (
    "time/fps", "rollout/ep_rew_mean", "rollout/ep_len_mean",
    "train/policy_gradient_loss", "train/value_loss", "train/entropy_loss",
    "train/approx_kl", "train/clip_fraction", "train/explained_variance",
    "train/learning_rate", "train/n_updates", "train/loss",
)
SAC_KEYS = (
    "time/fps", "rollout/ep_rew_mean", "rollout/ep_len_mean",
    "train/loss/policy", "train/loss/critic", "train/loss/alpha",
    "train/policy/alpha", "train/value/q_mean", "train/policy/log_pi_mean",
    "train/ent_coef", "train/learning_rate", "train/n_updates",
)


def _col_floats(rows, key):
    ys = []
    for row in rows:
        v = row.get(key)
        if v in (None, ""):
            ys.append(np.nan)
            continue
        try:
            ys.append(float(v))
        except (TypeError, ValueError):
            ys.append(np.nan)
    return np.asarray(ys, dtype=float)


def _series(rows, key):
    y = _col_floats(rows, key)
    if np.isfinite(y).any():
        return y
    for alt in _SAC_ALIASES.get(key, ()):
        y = _col_floats(rows, alt)
        if np.isfinite(y).any():
            return y
    return y


def _ma_nan(x, w=10):
    """Causal trailing mean (expanding until w), NaN-aware (iso-trading-bot port)."""
    y = np.asarray(x, dtype=float)
    if y.size == 0:
        return y.copy()
    w = max(1, int(w))
    if w == 1:
        return y.copy()
    ok = np.isfinite(y)
    y0 = np.where(ok, y, 0.0)
    cs_y = np.concatenate([[0.0], np.cumsum(y0)])
    cs_n = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    idx = np.arange(1, y.size + 1)
    j = np.maximum(0, idx - w)
    dn = cs_n[idx] - cs_n[j]
    out = np.full(y.size, np.nan)
    good = dn > 0
    out[good] = (cs_y[idx][good] - cs_y[j][good]) / dn[good]
    return out


def ml_health(csv_path, out_path, title="agent", theme="synthwave", ma=10):
    c = _palette(theme)
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    is_sac = "sac" in Path(csv_path).stem.lower()
    keys = SAC_KEYS if is_sac else PPO_KEYS
    titles = list(keys) + [""] * (12 - len(keys))
    xs = _col_floats(rows, "time/total_timesteps")
    if np.isfinite(xs).any():
        xlabel = "timesteps"
    else:
        xs = np.arange(len(rows), dtype=float)
        xlabel = "dump"

    fig = make_subplots(rows=4, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.10)
    for i, k in enumerate(keys):
        r, col = divmod(i, 3)
        r, col = r + 1, col + 1
        ys = _series(rows, k)
        mask = np.isfinite(xs) & np.isfinite(ys)
        x_f, y_f = xs[mask], ys[mask]
        if y_f.size == 0:
            fig.update_yaxes(title_text=k.split("/", 1)[-1], row=r, col=col)
            continue
        trend = _ma_nan(y_f, ma)
        mode = "lines+markers" if y_f.size < 80 else "lines"
        fig.add_trace(go.Scatter(x=x_f, y=y_f, mode=mode, name="raw", legendgroup="raw", showlegend=(i == 0),
                                 line=dict(color=c["violet"], width=1.5),
                                 marker=dict(size=5, color=c["violet"], opacity=0.85), opacity=0.9), r, col)
        fig.add_trace(go.Scatter(x=x_f, y=trend, mode="lines", name=f"MA{ma}", legendgroup="ma", showlegend=(i == 0),
                                 line=dict(color=c["cyan"], width=2.3)), r, col)
        fig.update_yaxes(title_text=k.split("/", 1)[-1], row=r, col=col)
        fig.update_xaxes(title_text=xlabel, row=r, col=col)

    fig.update_layout(**_base_layout(c, title=dict(text=title or Path(csv_path).stem, x=0.01, xanchor="left"),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        margin=dict(l=72, r=36, t=88, b=64)))
    fig.update_annotations(font=dict(size=11, color=c["lime"]))
    _style_axes(fig, 4, 3, c)
    _write_png(fig, out_path, width=1480, height=1180)
    return out_path


def _episode_offsets(cfg):
    """Resolve the test seed offsets for ``eval.episodes``.

    The first ``len(TEST_SEED_OFFSETS)`` episodes replay the locked final-test
    draws (offsets 1..8). Any episodes beyond that extend the sequence
    deterministically (offsets 9..N), so ``eval.episodes`` is not capped by the
    number of pre-registered draws. Each offset maps to a unique GBM bundle
    draw via ``seed = cfg.seed + offset``.
    """
    ev = dict(cfg.get("eval") or {})
    n_episodes = max(1, int(ev.get("episodes", 1)))
    locked = list(TEST_SEED_OFFSETS)
    if n_episodes <= len(locked):
        return locked[:n_episodes]
    return locked + list(range(len(locked) + 1, n_episodes + 1))


def _nonempty(a, init_bal=1000.0):
    a = np.asarray(a)
    if a.size:
        return a
    return np.array([init_bal, init_bal])


def _aggregate_episodes(episodes):
    """Combine per-episode raw dicts into the portfolio-level report result.

    The portfolio net curve is the mean of the account curves across
    episodes (index-aligned timelines); per-symbol equity is likewise
    averaged across episodes. Trades are merged and sorted by close time.
    """
    ledger = _sort_ledger([t for e in episodes for t in e["ledger"]])
    n_ep = max(len(episodes), 1)

    def _mean(values):
        return np.mean(np.stack(values), axis=0) if len(values) > 1 else values[0]

    agg_per_symbol = None
    seen = 0
    for e in episodes:
        ps = np.asarray(e.get("per_symbol"))
        if ps is None or ps.size == 0:
            continue
        agg_per_symbol = ps if agg_per_symbol is None else agg_per_symbol + ps
        seen += 1
    if agg_per_symbol is not None:
        agg_per_symbol = agg_per_symbol / seen

    roc_eps = [e for e in episodes
               if e.get("roc_realized") is not None and e.get("roc_coll") is not None]
    if len(roc_eps) == len(episodes) and roc_eps:
        T = len(episodes[0]["roc_realized"])
        tot_real = np.zeros(T, dtype=float)
        tot_coll = np.zeros(T, dtype=float)
        for e in roc_eps:
            r = np.asarray(e["roc_realized"]).reshape(-1)[:T]
            c = np.asarray(e["roc_coll"]).reshape(-1)[:T]
            tot_real += r
            tot_coll += c
        roc = np.where(tot_coll > 1e-6, tot_real / np.maximum(tot_coll, 1e-9), 0.0)
    else:
        roc_eps_fallback = [np.asarray(e["roc"]) for e in episodes if e.get("roc") is not None]
        if len(roc_eps_fallback) == len(episodes) and roc_eps_fallback:
            roc = np.mean(np.stack([_nonempty(v) for v in roc_eps_fallback]), axis=0)
        elif roc_eps_fallback:
            roc = roc_eps_fallback[0]
        else:
            roc = np.zeros(1)

    return {
        "ledger": ledger,
        "net": _mean([_nonempty(e["net"]) for e in episodes]),
        "gross": _mean([_nonempty(e["gross"]) for e in episodes]),
        "steps": np.asarray(episodes[0]["steps"]),
        "per_symbol": agg_per_symbol,
        "roc": roc,
        "episodes": [
            {"net": _nonempty(e["net"]), "gross": _nonempty(e["gross"]),
             "steps": np.asarray(e["steps"]), "roc": np.asarray(e.get("roc"))}
            for e in episodes
        ],
        "n_episodes": n_ep,
    }


def generate_report(cfg, manager, worker, out_dir, norm_state=None, deterministic=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    theme = resolve_theme((cfg.get("report") or {}).get("theme", "synthwave"))
    overlays = bool((cfg.get("report") or {}).get("overlays", True))
    ep_figs = bool((cfg.get("report") or {}).get("episode_figures", False))
    ret_basis = (cfg.get("returns") or {}).get("basis", "account")
    if deterministic is None:
        deterministic = bool((cfg.get("eval") or {}).get("deterministic", True))
    offsets = _episode_offsets(cfg)
    log.info("report: %d test episodes (seed offsets %s, theme=%s, overlays=%s, episode_figures=%s, returns=%s, deterministic=%s)",
             len(offsets), offsets, theme, overlays, ep_figs, ret_basis, deterministic)

    episodes = []
    per_ep_steps = max(int((cfg.get("data") or {}).get("n_steps", 400)) - 1, 1)
    pbar = tqdm(total=len(offsets) * per_ep_steps, desc="test phase",
                unit="step", colour="magenta", leave=True)
    for ep, off in enumerate(offsets):
        t_ep = time.monotonic()
        raw = run_test(cfg, manager, worker, norm_state=norm_state, seed_offset=off, pbar=pbar,
                       deterministic=deterministic)
        for t in raw["ledger"]:
            t["episode"] = ep
            t["seed_offset"] = off
        episodes.append(raw)
        log.info("report: episode %d/%d (seed%d) done in %.2fs: %d trades, eq=%.4f",
                 ep + 1, len(offsets), off, time.monotonic() - t_ep,
                 len(raw["ledger"]), float(np.asarray(raw["net"])[-1]))
        # Each episode builds a fresh GBM bundle + features on the Metal device;
        # release the MLX allocator cache between draws so long test phases do
        # not exhaust the Metal heap. ``raw`` holds only numpy arrays, so no
        # live tensors are dropped.
        import mlx.core as mx
        mx.clear_cache()
    pbar.close()

    result = _aggregate_episodes(episodes)
    log.info("report: %d episodes %d trades final_equity=%.4f",
             len(offsets), len(result["ledger"]),
             float(result["net"][-1]) if len(result["net"]) else 0.0)

    rets = None
    if ret_basis == "collateral" and np.asarray(result.get("roc")).size > 1:
        rets = np.asarray(result["roc"])[1:]
    t_figs = time.monotonic()
    write_ledger(result["ledger"], out_dir / "trades.csv")
    log.debug("report: trades.csv written in %.2fs (%d rows)",
              time.monotonic() - t_figs, len(result["ledger"]))
    breakdown(result, out_dir / "breakdown.txt", cfg, rets=rets, basis=ret_basis)
    log.debug("report: breakdown.txt written in %.2fs", time.monotonic() - t_figs)
    figure1(result, out_dir / "figure1.png", theme=theme, overlays=overlays, ret_series=rets)
    log.debug("report: figure1.png written in %.2fs", time.monotonic() - t_figs)
    figure2(result, out_dir / "figure2.png", theme=theme)
    log.debug("report: figure2.png written in %.2fs", time.monotonic() - t_figs)
    n_ep_figs = 0
    if ep_figs:
        for raw in episodes:
            per_ep = {
                "ledger": raw["ledger"],
                "net": _nonempty(raw["net"]),
                "gross": _nonempty(raw["gross"]),
                "steps": np.asarray(raw["steps"]),
                "per_symbol": np.asarray(raw.get("per_symbol"))
                if np.asarray(raw.get("per_symbol")).size else None,
            }
            ep_rets = None
            if ret_basis == "collateral" and np.asarray(raw.get("roc")).size > 1:
                ep_rets = np.asarray(raw["roc"])[1:]
            figure1(per_ep, out_dir / f"figure1_episode_{raw['seed_offset']}.png", theme=theme,
                    overlays=overlays, ret_series=ep_rets)
            figure2(per_ep, out_dir / f"figure2_episode_{raw['seed_offset']}.png", theme=theme)
            n_ep_figs += 2
    log.debug("report: %d episode figures written; figure phase %.2fs total",
              n_ep_figs, time.monotonic() - t_figs)
    log.debug("report artifacts -> %s", out_dir)
    agg_net = result["net"]
    m = metrics(agg_net, periods_per_year=_periods_per_year(cfg), rets=rets, basis=ret_basis)
    log.debug("report metrics: %s", m)
    return {"out_dir": out_dir, "metrics": m, "n_trades": len(result["ledger"])}
