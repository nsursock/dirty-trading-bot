"""Policy rollout + search objective; reporting is delegated to
`dirty-fin-reports <https://github.com/nsursock/dirty-fin-reports>`_.

This module keeps the pieces that only the bot can do — driving a trained
joint policy deterministically over a fresh synthetic test bundle and
reconstructing the trade ledger from the env state — plus the hyperparameter
search objective (``validate`` / ``dsr``). The reporting layer proper (risk
metrics, plausibility checks, verdicts, breakdown text, PNG figures and
``report.json``) is provided by the ``dirty_fin_reports`` package:

* ``run_test`` emits the closed-position ledger the package consumes;
* ``generate_report`` writes ``trades.csv`` and hands the run folder to
  ``dirty_fin_reports.simple.report.run_reporter``, which emits
  ``report.json``, ``breakdown.txt`` and the verdict-named PNGs.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from pathlib import Path

import numpy as np
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
        ar_coef=d.get("ar", 0.0),
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
    mtm_curve = []
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
                    "entry_conviction": float(abs(act_np[i])),
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
                    "entry_conviction": float(abs(act_np[i])),
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
        mtm_curve.append(float(np.mean(equity)))

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
        "mtm": np.asarray(mtm_curve),
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
        "trade_id", "episode", "seed_offset", "symbol", "side", "opened_at", "closed_at",
        "entry_price", "exit_price", "notional", "leverage", "collateral", "equity_before",
        "entry_conviction", "fee", "funding", "realized_pnl", "exit_type",
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


_FREQ_PPY: dict[str, float] = {
    "1m": 252 * 24 * 60,
    "5m": 252 * 24 * 12,
    "15m": 252 * 24 * 4,
    "30m": 252 * 24 * 2,
    "1h": 252 * 24,
    "4h": 252 * 6,
    "1d": 252,
    "daily": 252,
    "1w": 52,
    "weekly": 52,
}


def _resample_equity(net, periods_per_year, freq):
    """Aggregate a bar-indexed equity curve into ``freq`` simple returns.

    Equity is marked at the close of every ``freq`` window and the per-window
    return is ``end/start - 1``, so Sharpe / Sortino are computed on portfolio
    returns at the reporting cadence instead of per-bar steps (which are
    mostly flat plateaus between trade closes). Returns ``(None, ppy, net)``
    when ``freq`` is coarser than the data or there are fewer than two
    windows; otherwise ``(rets, ppy, levels)`` where ``levels`` is the equity
    marked at each window close (used for drawdown metrics).
    """
    fppy = _FREQ_PPY.get(freq)
    if fppy is None or fppy <= 0:
        raise ValueError(f"unknown reporting frequency: {freq!r}")
    bars_per_agg = max(round(periods_per_year / fppy), 1)
    if bars_per_agg <= 1:
        return None, periods_per_year, np.asarray(net, dtype=float)
    n = len(net)
    starts = net[np.arange(0, n - 1, bars_per_agg)]
    ends = net[np.minimum(np.arange(bars_per_agg, n, bars_per_agg), n - 1)]
    k = min(len(starts), len(ends))
    if k < 2:
        return None, periods_per_year, np.asarray(net, dtype=float)
    return ends[:k] / starts[:k] - 1.0, fppy, np.concatenate([starts[:1], ends[:k]])


def metrics(net, periods_per_year=252, rets=None, basis="account",
            rf_annual=0.0, freq="bar"):
    """Risk metrics on a single equity curve (search objective).

    This remains for the hyperparameter search objective (``validate`` /
    ``dsr``); the final report metrics come from ``dirty_fin_reports``.

    ``net`` must be the bar ``net_curve`` from ``run_test``. ``periods_per_year``
    is the native bar cadence from ``_periods_per_year(cfg)``.

    ``freq`` picks the reporting cadence for Sharpe / Sortino: ``"bar"`` keeps
    the native per-bar returns (legacy behavior), anything in ``_FREQ_PPY``
    (e.g. ``"daily"``, ``"1h"``) aggregates the equity curve to that frequency
    first — 5-minute bars annualized by ``sqrt(72576)`` report a Sharpe that
    ignores the flat plateaus between closes. ``rf_annual`` is subtracted as a
    per-period risk-free rate. When ``freq != "bar"`` the return series is
    derived from the equity curve (a true portfolio return), regardless of
    ``rets``; at ``freq="bar"`` ``rets`` (e.g. the collateral-basis ROC series)
    is used as-is.

    ``final_equity`` / ``total_return`` / ``max_drawdown`` / ``cagr`` stay on
    the dollar equity curve. ``sortino`` is ``None`` when the sample has fewer
    than two downside periods (or zero downside dispersion) — it is undefined,
    not a lucky huge number.

    ``ulcer_index`` is the root-mean-square of the percentage drawdowns over
    the reporting cadence: deep *or* prolonged holes both inflate it, so a
    curve with shallow, short dips scores low. ``upi`` (Ulcer Performance
    Index / Martin Ratio) is the annualized excess mean return divided by the
    Ulcer Index — the goal-state metric for a "healthy staircase" equity
    curve. It is ``None`` when the Ulcer Index is ~0 (the curve never left its
    running high, so the ratio is undefined, not infinite).
    """
    if rets is None or np.asarray(rets).size == 0:
        rets = _returns(net)
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 2 or np.std(rets) == 0:
        return {}
    ppy_bar = periods_per_year
    levels = np.asarray(net, dtype=float)
    if freq != "bar":
        agg, ppy, levels = _resample_equity(levels, periods_per_year, freq)
        if agg is not None:
            rets = agg
            periods_per_year = ppy
    excess = rets - rf_annual / periods_per_year
    mean, std = float(np.mean(excess)), float(np.std(excess))
    if std <= 1e-12:
        return {}
    down = excess[excess < 0]
    sortino = None
    if len(down) >= 2:
        dstd = float(np.std(down))
        if dstd > 1e-12:
            sortino = mean / (dstd + 1e-12) * np.sqrt(periods_per_year)
    peak = np.maximum.accumulate(levels)
    dd = (peak - levels) / peak
    ulcer_index = float(np.sqrt(np.mean(np.square(dd))))
    upi = None
    if ulcer_index > 1e-12:
        upi = mean * periods_per_year / ulcer_index
    peak_net = np.maximum.accumulate(net)
    dd_net = (peak_net - net) / peak_net
    years = len(net) / ppy_bar
    cagr = (net[-1] / net[0]) ** (1.0 / years) - 1.0 if years > 0 and net[0] > 0 else 0.0
    return {
        "sharpe": mean / (std + 1e-12) * np.sqrt(periods_per_year),
        "sortino": sortino,
        "max_drawdown": float(np.max(dd_net)),
        "ulcer_index": ulcer_index,
        "upi": upi,
        "cagr": cagr,
        "final_equity": float(net[-1]),
        "total_return": float(net[-1] / net[0] - 1.0),
        "return_basis": basis,
        "freq": freq,
        "rf_annual": rf_annual,
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


def resolve_theme(theme: str) -> str:
    """Resolve a theme name via the reporting engine (dirty-fin-reports)."""
    from dirty_fin_reports.simple.viz import resolve_theme as _resolve

    return _resolve(theme)


def _report_config(cfg):
    """Map the bot config onto the reporting engine's ``ReportConfig``."""
    from dirty_fin_reports.simple.config import ReportConfig

    d = cfg.get("data") or {}
    e = cfg.get("env") or {}
    r = cfg.get("returns") or {}
    rep = cfg.get("report") or {}
    tf = (d.get("timeframes") or {}).get("low", "5m")
    return ReportConfig(
        timeframe=str(tf),
        initial_balance=float(e.get("initial_balance", 1000.0)),
        reporting_freq=str(r.get("freq", "daily")),
        rf_annual=float(r.get("rf_annual", 0.045)),
        n_steps=int(d.get("n_steps", 400)),
        start_date=rep.get("start_date"),
        tick_tilt=bool(rep.get("tick_tilt", True)),
        tick_angle=float(rep.get("tick_angle", 22.5)),
        tick_direction=str(rep.get("tick_direction", "down")),
    )


def _plausibility(cfg):
    """Build a ``Plausibility`` bounds object from the bot config (defaults if absent)."""
    from dirty_fin_reports.simple.config import Plausibility

    raw = (cfg.get("report") or {}).get("plausibility") or {}
    if not raw:
        return Plausibility()
    bounds = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            bounds[k] = (None if v[0] is None else float(v[0]),
                         None if v[1] is None else float(v[1]))
    return Plausibility(**bounds)


def _flatten_accounts(ledger):
    """Key each (episode, symbol) account with a unique integer ``episode``.

    The reporting engine reconstructs the portfolio as the mean of one account
    per ``episode``; the bot runs one account per symbol per episode, so each
    account gets ``episode = ep * n_symbols + symbol_index`` and the real
    test-episode draw stays in ``seed_offset``.
    """
    syms = sorted({str(t.get("symbol") or "") for t in ledger})
    sym_index = {s: i for i, s in enumerate(syms)}
    out = []
    for t in ledger:
        t = dict(t)
        ep = int(t.get("episode", 0) or 0)
        t["episode"] = ep * len(syms) + sym_index[str(t.get("symbol") or "")]
        out.append(t)
    return out


def generate_report(cfg, manager, worker, out_dir, norm_state=None, deterministic=None):
    """Rollout the policy, write ``trades.csv``, and delegate reporting.

    The policy rollout is the only bot-specific step left here: it runs the
    trained joint policy deterministically over the locked test bundle, tags
    each closed trade with its episode / seed offset, and writes the ledger
    CSV. Everything else (risk metrics, plausibility checks, verdict,
    ``breakdown.txt``, PNG figures, ``report.json`` and agent diagnostics) is
    produced by ``dirty_fin_reports.simple.report.run_reporter``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = cfg.get("report") or {}
    theme = resolve_theme(rep.get("theme", "synthwave"))
    overlays = bool(rep.get("overlays", False))
    if deterministic is None:
        deterministic = bool((cfg.get("eval") or {}).get("deterministic", True))
    offsets = _episode_offsets(cfg)
    log.info("report: %d test episodes (seed offsets %s, theme=%s, overlays=%s, deterministic=%s)",
             len(offsets), offsets, theme, overlays, deterministic)

    all_trades = []
    per_ep_steps = max(int((cfg.get("data") or {}).get("n_steps", 400)) - 1, 1)
    pbar = tqdm(total=len(offsets) * per_ep_steps, desc="test phase",
                unit="step", colour="magenta", leave=True)
    for ep, off in enumerate(offsets):
        t_ep = time.monotonic()
        raw = run_test(cfg, manager, worker, norm_state=norm_state, seed_offset=off,
                       pbar=pbar, deterministic=deterministic)
        for t in raw["ledger"]:
            t["episode"] = ep
            t["seed_offset"] = off
        all_trades.extend(raw["ledger"])
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

    ledger = _flatten_accounts(_sort_ledger(all_trades))
    write_ledger(ledger, out_dir / "trades.csv")

    run_root = out_dir.parent
    from dirty_fin_reports.simple.report import run_reporter

    report = run_reporter(
        run_root,
        out_dir=run_root,
        config=_report_config(cfg),
        theme=theme,
        overlays=overlays,
        plausibility=_plausibility(cfg),
        meta={"data_origin": "policy_rollout"},
    )
    verdict = report["plausibility"]
    log.info("report: %d trades -> verdict=%s (%s) recommendation=%s",
             report["ledger"]["n_trades"], verdict["status"], verdict["counts"],
             report["recommendation"]["action"])
    return {
        "out_dir": out_dir,
        "metrics": report["portfolio"],
        "n_trades": report["ledger"]["n_trades"],
        "report": report,
    }
