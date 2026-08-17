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
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate

from data import SYMBOLS, generate
from env import TradingEnv

log = logging.getLogger("trading")


def _np(x) -> np.ndarray:
    return np.asarray(x.tolist()) if hasattr(x, "tolist") else np.asarray(x)


def _one_hot(a, n):
    import mlx.core as mx

    return mx.equal(mx.arange(n), a[:, None]).astype(mx.float32)


def _test_env(cfg, seed_offset=1):
    import mlx.core as mx

    d = dict(cfg.get("data", {}))
    e = dict(cfg.get("env", {}))
    r = dict(cfg.get("reward", {}))
    h = dict(cfg.get("hrl", {}))
    e["n_envs_per_symbol"] = 1
    e["reward_mode"] = r.get("mode", "smoke")
    e["drawdown_penalty"] = r.get("drawdown_penalty", 1.0)
    e["goal_dim"] = h.get("goal_dim", 3)
    e["action_space"] = "continuous"
    symbols = dict(list(SYMBOLS.items())[: d.get("n_symbols", 4)])
    seed = cfg.get("seed", 42) + seed_offset
    bundle = generate(
        symbols=symbols,
        n_steps=d.get("n_steps", 400),
        seed=seed,
        dt=1.0 / d.get("dt_days", 252),
    )
    env = TradingEnv(bundle.features, bundle.ohlcv.closes, seed=seed, **e)
    return env, list(bundle.symbols), e


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


def run_test(cfg, manager, worker) -> dict:
    """Deterministic joint rollout -> ledger + per-step series."""
    import mlx.core as mx

    env, symbols, e = _test_env(cfg)
    log.info("test: env num_envs=%d n_symbols=%d T=%d", env.num_envs, env.n_symbols, env.T)
    goal_every = cfg.get("hrl", {}).get("goal_every", 4)
    goal_dim = cfg.get("hrl", {}).get("goal_dim", 3)
    obs_mgr_dim = env.F + 6
    F, init_bal = env.F, e.get("initial_balance", 1000.0)
    n_envs = env.num_envs
    sym_idx = [s for s in range(env.n_symbols) for _ in range(env.n_envs_per_symbol)]
    side_thr = e.get("side_threshold", 0.2)
    fee_rate = e.get("fee_rate", 5e-4)
    funding = e.get("funding_rate", 1e-4)
    T = env.T

    worker_obs = env.reset()[0]
    mgr_obs = worker_obs[:, :obs_mgr_dim]

    open_trades = [None] * n_envs
    ledger = []
    prev_side = np.zeros(n_envs, dtype=int)
    cum_fee = np.zeros(n_envs)

    net_curve, gross_curve, step_axis = [], [], []
    lev_all, coll_all, sides_all, exits = [], [], [], []
    trade_id = 0

    t = 0
    while t < T - 1:
        goal, _, _ = manager.policy.get_action(mgr_obs, deterministic=True)
        env.set_goal(_one_hot(goal, goal_dim))
        for _ in range(goal_every):
            if t >= T - 1:
                break
            act = worker._scale_action(worker.actor.sample(worker_obs, deterministic=True)[0])
            act_np = _np(act).reshape(-1)
            worker_obs, r, done, info = env.step(act)
            t += 1

            state = env._state
            q = _np(state[:, 1])
            entry = _np(state[:, 2])
            collateral = _np(state[:, 3])
            balance = _np(state[:, 0])
            t_idx = np.minimum(_np(env._steps), T - 1)
            price = _np(mx.take(env.closes_flat, env.sym_off + mx.array(t_idx, mx.int32)))
            equity = balance + collateral + q * (price - entry)
            notional = np.abs(q) * price
            side = np.sign(q).astype(int)
            req = np.where(np.abs(act_np) > side_thr, np.sign(act_np), 0).astype(int)

            for i in range(n_envs):
                if prev_side[i] == 0 and side[i] != 0:
                    open_trades[i] = {
                        "trade_id": trade_id,
                        "symbol": symbols[sym_idx[i]],
                        "side": "long" if side[i] > 0 else "short",
                        "opened_at": t,
                        "entry_price": float(price[i]),
                        "notional": float(notional[i]),
                        "leverage": float(notional[i] / (collateral[i] + 1e-9)),
                        "equity_before": float(equity[i]),
                        "fee": fee_rate * notional[i],
                        "exit_type": "open",
                    }
                    trade_id += 1
                    cum_fee[i] += fee_rate * notional[i]
                elif prev_side[i] != 0 and side[i] == 0:
                    tr = open_trades[i]
                    realized = float(equity[i] - tr["equity_before"])
                    exit_type = "market_close" if req[i] == 0 else "liquidation"
                    if tr is not None:
                        ledger.append(
                            _finish_trade(tr, float(price[i]), t, exit_type, fee_rate * notional[i], realized)
                        )
                    open_trades[i] = None
                    exits.append(exit_type)
                    cum_fee[i] += fee_rate * abs(tr["notional"]) if tr else 0.0
                elif prev_side[i] != 0 and side[i] != prev_side[i]:
                    tr = open_trades[i]
                    if tr is not None:
                        realized = float(equity[i] - tr["equity_before"])
                        ledger.append(
                            _finish_trade(tr, float(price[i]), t, "market_close", fee_rate * notional[i], realized)
                        )
                        exits.append("market_close")
                    open_trades[i] = {
                        "trade_id": trade_id,
                        "symbol": symbols[sym_idx[i]],
                        "side": "long" if side[i] > 0 else "short",
                        "opened_at": t,
                        "entry_price": float(price[i]),
                        "notional": float(notional[i]),
                        "leverage": float(notional[i] / (collateral[i] + 1e-9)),
                        "equity_before": float(equity[i]),
                        "fee": fee_rate * notional[i],
                        "exit_type": "open",
                    }
                    trade_id += 1

                cum_fee[i] += funding * notional[i]
                if side[i] != 0:
                    lev_all.append(notional[i] / (collateral[i] + 1e-9))
                    coll_all.append(collateral[i])
                    sides_all.append("long" if side[i] > 0 else "short")

            prev_side = side
            step_axis.append(t)
            net_curve.append(float(np.mean(equity)))
            gross_curve.append(float(np.mean(equity + cum_fee)))

    log.debug("test rollout done: steps=%d trades=%d final_equity=%.4f", t, len(ledger), net_curve[-1])
    return {
        "ledger": ledger,
        "net": np.asarray(net_curve),
        "gross": np.asarray(gross_curve),
        "steps": np.asarray(step_axis),
        "leverage": np.asarray(lev_all),
        "collateral": np.asarray(coll_all),
        "sides": sides_all,
        "exits": exits,
        "env": env,
    }


def write_ledger(ledger, path):
    cols = [
        "trade_id", "symbol", "side", "opened_at", "closed_at", "entry_price",
        "exit_price", "notional", "leverage", "fee", "realized_pnl", "exit_type",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for t in ledger:
            w.writerow(t)
    return path


def _returns(equity):
    return np.diff(equity) / (equity[:-1] + 1e-12)


def metrics(net, periods_per_year=252):
    rets = _returns(net)
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
    }


def breakdown(result, path):
    ledger = result["ledger"]
    m = metrics(result["net"])
    lines = []
    lines.append("Table 1 — Performance by symbol")
    by_sym = {}
    for t in ledger:
        b = by_sym.setdefault(t["symbol"], {"pnl": 0.0, "n": 0, "liq": 0})
        b["pnl"] += t.get("realized_pnl", 0.0)
        b["n"] += 1
        b["liq"] += t.get("exit_type") == "liquidation"
    lines.append(tabulate(
        [[s, b["n"], round(b["pnl"], 4), b["liq"]] for s, b in sorted(by_sym.items())],
        headers=["symbol", "trades", "pnl", "liquidations"],
        tablefmt="github",
    ))
    lines.append("")
    lines.append("Table 2 — Aggregate metrics")
    lines.append(tabulate([[k, round(v, 4)] for k, v in m.items()], headers=["metric", "value"], tablefmt="github"))
    lines.append("")
    lines.append("Table 3 — Exit type counts")
    exits = result["exits"]
    lines.append(tabulate(
        [[e, exits.count(e)] for e in sorted(set(exits))],
        headers=["exit_type", "count"],
        tablefmt="github",
    ))
    text = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(text)
    return text


def figure1(result, path):
    net, gross = result["net"], result["gross"]
    steps = result["steps"]
    rets = _returns(net)
    peak = np.maximum.accumulate(net)
    dd = (peak - net) / peak

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Equity curves", "Returns over time", "Drawdown", "Return distribution"))
    fig.add_trace(go.Scatter(x=steps, y=net, name="net"), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=gross, name="gross"), 1, 1)
    fig.add_trace(go.Scatter(x=steps[1:], y=rets, name="returns"), 1, 2)
    fig.add_trace(go.Scatter(x=steps, y=-dd, name="drawdown"), 2, 1)
    fig.add_trace(go.Histogram(x=rets, nbinsx=50, name="ret dist"), 2, 2)
    fig.update_layout(title="Figure 1 — Performance", height=800)
    fig.write_image(path, scale=2.0)
    return path


def figure2(result, path):
    lev, coll = result["leverage"], result["collateral"]
    sides, exits = result["sides"], result["exits"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Leverage distribution", "Collateral distribution", "Long/Short", "Exit types"))
    fig.add_trace(go.Histogram(x=lev, nbinsx=40), 1, 1)
    fig.add_trace(go.Histogram(x=coll, nbinsx=40), 1, 2)
    from collections import Counter

    fig.add_trace(go.Bar(x=list(Counter(sides)), y=list(Counter(sides).values())), 2, 1)
    fig.add_trace(go.Bar(x=list(Counter(exits)), y=list(Counter(exits).values())), 2, 2)
    fig.update_layout(title="Figure 2 — Risk & behavior", height=800)
    fig.write_image(path, scale=2.0)
    return path


def _csv_columns(path):
    with open(path) as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    if not rows:
        return {}, np.array([])
    cols = list(rows[0].keys())
    data = {c: np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows]) for c in cols}
    return data, cols


def _ma(x, n=10):
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.nan_to_num(x))
    for i in range(len(x)):
        lo = max(0, i - n + 1)
        out[i] = c[i] - (c[lo - 1] if lo > 0 else 0)
        out[i] /= (i - lo + 1)
    return out


def ml_health(csv_path, out_path, title="agent"):
    data, _ = _csv_columns(csv_path)
    if not data:
        return None
    picks = [c for c in data if any(
        k in c for k in ("ep_rew_mean", "value_loss", "policy_gradient", "actor_loss",
                         "critic_loss", "ent_coef", "entropy_loss", "clip_fraction",
                         "approx_kl", "explained_variance", "fps", "loss", "q_mean", "log_pi")
    )][:12]
    fig = make_subplots(rows=4, cols=3, subplot_titles=picks)
    for i, c in enumerate(picks):
        r, col = divmod(i, 3)
        fig.add_trace(go.Scatter(y=data[c], mode="lines", name=c), r + 1, col + 1)
        fig.add_trace(go.Scatter(y=_ma(data[c]), mode="lines", name=c + " (ma)"), r + 1, col + 1)
    fig.update_layout(title=f"ML health — {title}", height=1200)
    fig.write_image(out_path, scale=2.0)
    return out_path


def generate_report(cfg, manager, worker, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("report: deterministic test rollout")
    result = run_test(cfg, manager, worker)
    log.info("report: %d trades, final_equity=%.4f", len(result["ledger"]), float(result["net"][-1]))
    write_ledger(result["ledger"], out_dir / "trades.csv")
    breakdown(result, out_dir / "breakdown.txt")
    figure1(result, out_dir / "figure1.png")
    figure2(result, out_dir / "figure2.png")
    log.debug("report artifacts -> %s", out_dir)
    m = metrics(result["net"])
    log.debug("report metrics: %s", m)
    return {"out_dir": out_dir, "metrics": m, "n_trades": len(result["ledger"])}
