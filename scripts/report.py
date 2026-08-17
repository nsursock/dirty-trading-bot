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
from collections import Counter
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
    tf = d.get("timeframes", {})
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
        low_tf=tf.get("low", 5),
        high_tf=tf.get("high", 240),
        regime=d.get("regime", "bull"),
    )
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

    high_idx = mx.minimum(low_steps // n_resample, T_hi - 1)
    feats = mx.take(high_feats, sym_off_hi + high_idx, axis=0)
    acct = worker_obs[:, F : F + 6]
    return mx.concatenate([feats, acct], axis=1)


def run_test(cfg, manager, worker, norm_state=None) -> dict:
    """Deterministic joint rollout -> ledger + per-step series."""
    import mlx.core as mx
    from dirty_mlx_ml.reinforcement import VecNormalize

    bundle, env, symbols, e = _test_env(cfg)
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
    high_feats = mx.reshape(bundle.high_features, (S * T_hi, bundle.high_features.shape[2]))
    side_thr = e.get("side_threshold", 0.2)
    fee_rate = e.get("fee_rate", 5e-4)
    funding = e.get("funding_rate", 1e-4)
    T = env.T

    worker_obs = env.reset()[0]
    mgr_obs = _mgr_obs_test(high_feats, sym_off_hi, worker_obs, mx.zeros((n_envs,), mx.int32),
                            T_hi, F, n_resample)

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
        for _k in range(goal_every):
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
            exit_flags = _np(info.get("exit", mx.zeros((n_envs,), mx.int32))).astype(int)
            _EXIT_NAMES = {0: "market_close", 1: "take_profit", 2: "stop_loss", 3: "liquidation"}

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
                    exit_type = _EXIT_NAMES.get(int(exit_flags[i]),
                                                "market_close" if req[i] == 0 else "liquidation")
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

            if _k == goal_every - 1:
                low_steps = mx.minimum(env._steps, T - 1) if hasattr(env, "_steps") else t
                mgr_obs = _mgr_obs_test(high_feats, sym_off_hi, worker_obs, low_steps, T_hi, F, n_resample)

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


_BD_COLS = ["label", "num trades", "win rate %", "avg win", "avg loss",
            "net profit", "sharpe", "max dd", "risk reward", "sortino", "calmar", "profit factor"]


def _config_lines(cfg) -> list[str]:
    d = cfg.get("data", {})
    e = cfg.get("env", {})
    r = cfg.get("reward", {})
    h = cfg.get("hrl", {})
    w = cfg.get("worker", {})
    return [
        f"seed: {cfg.get('seed')}",
        f"symbols: {d.get('n_symbols')}  steps: {d.get('n_steps')}  dt_days: {d.get('dt_days')}",
        f"env: {d.get('n_symbols')} symbols x {e.get('n_envs_per_symbol')} = {d.get('n_symbols',0)*e.get('n_envs_per_symbol',0)} envs  "
        f"lev {e.get('lev_min')}–{e.get('lev_max')}x  risk {float(e.get('risk_min',0))*100:.0f}–{float(e.get('risk_max',0))*100:.0f}%  "
        f"eq={e.get('initial_balance')}  fee={e.get('fee_rate')}  funding={e.get('funding_rate')}",
        f"reward: {r.get('mode')}  dd_pen={r.get('drawdown_penalty')}  clip={r.get('reward_clip')}  trade_knob={e.get('trade_knob')}",
        f"hrl: goal_every={h.get('goal_every')}  goal_dim={h.get('goal_dim')}  "
        f"manager n_steps={cfg.get('manager',{}).get('n_steps')}  worker net={w.get('net_arch')} lr={w.get('learning_rate')}",
    ]


def _trade_stats(trades, base=1000.0):
    pnls = np.array([float(t.get("realized_pnl", 0.0) or 0.0) for t in trades], dtype=float)
    n = int(pnls.size)
    if n == 0:
        return dict(num=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0, net=0.0, sharpe=0.0,
                    max_dd=0.0, rr=0.0, sortino=0.0, calmar=0.0, pf=0.0)
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    net = float(pnls.sum())
    mean = float(pnls.mean())
    std = float(pnls.std())
    win_rate = 100.0 * float((pnls > 0).mean())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    sharpe = mean / std * np.sqrt(n) if std > 1e-12 else 0.0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    dd_min = float(dd.min())
    max_dd = 100.0 * abs(dd_min) / max(abs(base), 1.0) if abs(dd_min) > 1e-9 else 0.0
    rr = avg_win / abs(avg_loss) if losses.size and abs(avg_loss) > 1e-12 else 0.0
    down = pnls[pnls < 0]
    dstd = float(down.std()) if down.size else 0.0
    sortino = mean / dstd * np.sqrt(n) if dstd > 1e-12 else 0.0
    gw = float(wins.sum()) if wins.size else 0.0
    gl = float(abs(losses.sum())) if losses.size else 0.0
    pf = gw / gl if gl > 1e-12 else (999.0 if gw > 0 else 0.0)
    calmar = mean / abs(dd_min) if abs(dd_min) > 1e-12 else 0.0
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
    return [title, "", tabulate(rows, headers="firstrow", tablefmt="github", floatfmt=".4f"), ""]


def breakdown(result, path, cfg=None):
    ledger = result["ledger"]
    net = np.asarray(result["net"])
    base = float(net[0]) if net.size else 1000.0
    m = metrics(net)
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

    by_side = _bucket(ledger, lambda t: t["side"], ["long", "short"])
    by_exit = _bucket(ledger, lambda t: t["exit_type"],
                      ["take_profit", "stop_loss", "market_close", "liquidation"])
    by_outcome = _bucket(ledger, lambda t: ("win" if (t.get("realized_pnl", 0) or 0) > 0 else
                                            "loss" if (t.get("realized_pnl", 0) or 0) < 0 else "breakeven"),
                         ["win", "loss", "breakeven"])

    def _lev(t):
        v = float(t.get("leverage", 0) or 0)
        return "0-2.5x" if v <= 2.5 else "2.5-5x" if v <= 5 else "5-7.5x" if v <= 7.5 else "7.5-10x" if v <= 10 else "10x+"
    by_lev = _bucket(ledger, _lev, ["0-2.5x", "2.5-5x", "5-7.5x", "7.5-10x", "10x+"])

    def _hold(t):
        d = int(t.get("closed_at", 0) or 0) - int(t.get("opened_at", 0) or 0)
        return "flash (<2)" if d < 2 else "scalp (2-5)" if d < 5 else "sprint (5-15)" if d < 15 else "sit (15-60)" if d < 60 else "camp (60+)"
    by_hold = _bucket(ledger, _hold, ["flash (<2)", "scalp (2-5)", "sprint (5-15)", "sit (15-60)", "camp (60+)"])

    def _roe(t):
        eb = float(t.get("equity_before", 0) or 0)
        r = 100.0 * (float(t.get("realized_pnl", 0) or 0)) / max(abs(eb), 1e-9)
        return "multi-R (>=10%)" if r >= 10 else "single-R (1-10%)" if r >= 1 else "scratch (<1%)" if r >= 0 else "loss (<0%)"
    by_roe = _bucket(ledger, _roe, ["scratch (<1%)", "single-R (1-10%)", "multi-R (>=10%)", "loss (<0%)"])

    def _notional(t):
        v = float(t.get("notional", 0) or 0)
        return "toy (<1k)" if v < 1000 else "standard (1-10k)" if v < 10_000 else "size (10-50k)" if v < 50_000 else "whale (>=50k)"
    by_notional = _bucket(ledger, _notional, ["toy (<1k)", "standard (1-10k)", "size (10-50k)", "whale (>=50k)"])

    def _fee_drag(t):
        notional = max(abs(float(t.get("notional", 0) or 0)), 1e-9)
        bps = 10_000.0 * float(t.get("fee", 0) or 0) / notional
        return "free (maker)" if bps <= 0 else "light (<5 bps)" if bps < 5 else "heavy (>=5 bps)"
    by_fee = _bucket(ledger, _fee_drag, ["free (maker)", "light (<5 bps)", "heavy (>=5 bps)"])

    def _vintage(t):
        i = ledger.index(t)
        frac = i / n
        return "opening act (first 20%)" if frac < 0.2 else "encore (last 20%)" if frac >= 0.8 else "mid-set (20-80%)"
    by_vintage = _bucket(ledger, _vintage, ["opening act (first 20%)", "mid-set (20-80%)", "encore (last 20%)"])

    lines = ["BREAKDOWN", "=========", ""]
    if cfg is not None:
        lines += _config_lines(cfg)
        lines.append("")
    lines.append(f"portfolio: {port['num']} trades  final_equity={float(net[-1]):.2f}  "
                 f"ret={m.get('total_return', 0):+.2%}  sharpe={port['sharpe']:.3f}  max_dd={port['max_dd']:.2f}%")
    lines.append("")
    for title, groups in [
        ("By symbol", sym_groups),
        ("By side", by_side),
        ("By exit", by_exit),
        ("By outcome", by_outcome),
        ("By leverage", by_lev),
        ("By hold duration", by_hold),
        ("By RoE", by_roe),
        ("By notional", by_notional),
        ("By fee drag", by_fee),
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


def _palette(theme: str) -> dict:
    """Map a dirty-mkt-data theme (synthwave / ghibli / valorant) onto the
    iso-trading-bot ``_C`` palette roles (up=positive, down=negative, accent=highlight).

    ``muted`` (tick labels / subtitles) is derived from the background's
    brightness so it always has readable contrast on light *and* dark themes.
    """
    from dirty_mkt_data.viz.themes import THEMES

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


def figure1(result, path, theme="synthwave"):
    c = _palette(theme)
    net, gross = np.asarray(result["net"]), np.asarray(result["gross"])
    steps = np.asarray(result["steps"])
    rets = _returns(net)
    if rets.size == 0:
        rets = np.zeros(1)
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
    # glow under net
    fig.add_trace(go.Scatter(x=steps, y=net, mode="lines", line=dict(color=c["cyan"], width=7),
                             opacity=0.2, showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=net, mode="lines", line=dict(color=c["cyan"], width=2.6), name="Net"), 1, 1)
    fig.add_trace(go.Scatter(x=steps, y=gross, mode="lines", line=dict(color=c["lime"], width=1.8, dash="dot"), name="Gross"), 1, 1)
    fig.add_hline(y=float(net[0]), line=dict(color=c["muted"], width=1, dash="dash"), row=1, col=1)

    win = rets >= 0
    fig.add_trace(go.Scatter(x=steps[1:][win], y=rets[win], mode="markers",
                             marker=dict(size=7, color=c["cyan"], opacity=0.75, line=dict(width=0)), name="Win"), 1, 2)
    fig.add_trace(go.Scatter(x=steps[1:][~win], y=rets[~win], mode="markers",
                             marker=dict(size=7, color=c["magenta"], opacity=0.75, line=dict(width=0)), name="Loss"), 1, 2)
    fig.add_hline(y=0, line=dict(color=c["spine"], width=1), row=1, col=2)

    fig.add_trace(go.Scatter(x=steps, y=dd, mode="lines", fill="tozeroy", fillcolor=c["magenta_soft"],
                             line=dict(color=c["magenta"], width=2.4), name="Drawdown", showlegend=False), 2, 1)
    fig.add_trace(go.Histogram(x=rets, nbinsx=36, marker=dict(color=c["cyan"], line=dict(color=c["bg"], width=0.6), opacity=0.9),
                               showlegend=False), 2, 2)
    fig.add_vline(x=0, line=dict(color=c["muted"], width=1, dash="dash"), row=2, col=2)
    fig.add_vline(x=float(np.mean(rets)), line=dict(color=c["lime"], width=1.8), row=2, col=2)

    fig.update_layout(**_base_layout(c, title=dict(
        text=(f"Equity & risk<br><sup style='color:{c['muted']}'>"
              f"Net {net_ret:+.1%} · gross {gross_ret:+.1%} · costs {fees:,.0f} · max DD {max_dd:.1%} · {n_close} closes</sup>"),
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


def _ledger_equity(ledger, initial_balance=1000.0):
    """Reconstruct the (closed-trade) equity curves from the ledger only."""
    trades = sorted(ledger, key=lambda t: float(t.get("closed_at", 0) or 0))
    pnl = np.array([float(t.get("realized_pnl", 0.0) or 0.0) for t in trades])
    fees = np.array([float(t.get("fee", 0.0) or 0.0) for t in trades])
    equity = initial_balance + np.cumsum(pnl)
    gross = initial_balance + np.cumsum(pnl + fees)
    xs = np.array([float(t.get("closed_at", 0) or 0) for t in trades])
    return equity, gross, xs


def generate_report(cfg, manager, worker, out_dir, norm_state=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    theme = (cfg.get("report") or {}).get("theme", "synthwave")
    log.info("report: deterministic test rollout (theme=%s)", theme)
    raw = run_test(cfg, manager, worker, norm_state=norm_state)
    ledger = raw["ledger"]
    init_bal = float((cfg.get("env") or {}).get("initial_balance", 1000.0))
    net, gross, xs = _ledger_equity(ledger, init_bal)
    result = {
        "ledger": ledger,
        "net": net,
        "gross": gross,
        "steps": xs,
    }
    log.info("report: %d trades, final_equity=%.4f", len(ledger), float(net[-1]) if len(net) else init_bal)
    write_ledger(ledger, out_dir / "trades.csv")
    breakdown(result, out_dir / "breakdown.txt", cfg)
    figure1(result, out_dir / "figure1.png", theme=theme)
    figure2(result, out_dir / "figure2.png", theme=theme)
    log.debug("report artifacts -> %s", out_dir)
    m = metrics(net)
    log.debug("report metrics: %s", m)
    return {"out_dir": out_dir, "metrics": m, "n_trades": len(ledger)}
