"""AR signal-strength sweep.

Trains + evaluates the HRL bot once per ``phi`` (AR(1) coefficient) on
AR-injected GBM data. Every run shares the same template config (default
``configs/exp_ar.yaml``) with only ``data.ar`` changed, so the sole varying
quantity is the strength of the injected momentum signal.

    caffeinate -dims venv/bin/python utils/sweep_ar.py \
        --config configs/exp_ar_lev150.yaml \
        --phis -0.3,0,0.1,0.2,0.35,0.5,0.7 --timesteps 750000

Each phi run produces a timestamped run dir under ``logs/<ts>-<pid>/`` (data,
checkpoint, figures) exactly like a manual ``main.py full``. The sweep writes
``logs/sweeps/ar_sweep_<ts>.csv`` (one row per phi) and, unless ``--no-plot``,
``ar_sweep_<ts>.png`` plotting total_return/sharpe against the *measured*
lag-1 autocorrelation of the generated data.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "logs" / "sweeps"

_OUT_RE = re.compile(r"-> (?P<out>\S+)")


def _measured_ar1(phi: float, seed: int, ar_noise: float = 0.0) -> tuple[float, float]:
    """Lag-1 autocorr + per-step return std from the same bundle path as training."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from data import generate

    b = generate(
        symbols={"BTC": (0.30, 0.60, 60_000.0)},
        n_steps=20000, seed=seed, low_tf=5, high_tf=5, regime="neutral",
        ar_coef=phi,
        ar_noise=float(ar_noise),
    )
    r = np.log(np.asarray(b.ohlcv.closes))
    r = np.diff(r, axis=1)
    z = (r - r.mean(axis=-1, keepdims=True)) / (r.std(axis=-1, keepdims=True) + 1e-12)
    rho1 = float(np.mean(z[:, :-1] * z[:, 1:], axis=-1).item())
    return rho1, float(r.std(axis=-1).mean().item())


def _trade_stats(out_dir: Path) -> dict:
    p = out_dir / "trades.csv"
    rows = list(csv.DictReader(open(p)))
    n = len(rows)
    if n == 0:
        return {"trades": 0, "winrate": 0.0, "gross_pnl": 0.0, "liq": 0}
    pnl = [float(t.get("realized_pnl") or 0.0) for t in rows]
    wins = sum(1 for x in pnl if x > 1e-9)
    liq = sum(1 for t in rows if t.get("exit_type") == "liquidation")
    return {"trades": n, "winrate": wins / n, "gross_pnl": float(np.sum(pnl)), "liq": liq}


def _load_report(out_dir: Path) -> dict:
    """Read the reporting engine's ``report.json`` (next to the ``testing/`` dir)."""
    import json

    p = out_dir.parent / "report.json"
    if not p.exists():
        raise RuntimeError(f"no report.json next to {out_dir}")
    with open(p) as fh:
        return json.load(fh)


def _run_one(phi: float, timesteps: int, template: Path, out: Path, seed: int) -> dict:
    cfg = yaml.safe_load(template.read_text())
    cfg["data"]["ar"] = float(phi)
    cfg["seed"] = int(seed)
    cfg["eval"]["deterministic"] = True
    tag = f"phi_{phi:+.2f}_seed{seed}"
    cfg_path = out / f"{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    log_path = out / f"{tag}.log"
    # stderr inherited so main.py train/test tqdm bars render live; stdout
    # captured for the ``done ->`` line (pipe deadlock avoided — no capture on stderr).
    proc = subprocess.Popen(
        [sys.executable, "scripts/main.py", "full", "--config", str(cfg_path),
         "--timesteps", str(timesteps)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=None, text=True,
    )
    stdout, _ = proc.communicate()
    log_path.write_text(stdout or "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"phi={phi}: main.py exited {proc.returncode}\n{(stdout or '')[-1500:]}"
        )
    log_text = stdout or ""
    m = _OUT_RE.search(log_text)
    if m is None:
        raise RuntimeError(
            f"phi={phi}: no output dir line in {log_path}\n{log_text[-1500:]}"
        )
    out_dir = Path(m.group("out"))
    pm = _load_report(out_dir)["portfolio"]
    ts = _trade_stats(out_dir)

    def _f(key):
        v = pm.get(key)
        return float("nan") if v is None else float(v)

    row = {
        "phi": float(phi),
        "seed": int(seed),
        "total_return": float(pm.get("total_return") or 0.0),
        "sharpe": _f("sharpe"),
        "sortino": _f("sortino"),
        "max_drawdown": float(pm.get("max_drawdown") or 0.0),
        "ulcer_index": _f("ulcer_index"),
        "upi": _f("upi"),
        "final_equity": float(pm.get("final_equity") or 0.0),
        "cagr": float(pm.get("cagr") or 0.0),
        **ts,
    }
    # The whole sweep runs against ONE locked operating config
    # (min_hold_bars + side_threshold) so phi is the only varying quantity.
    # This is NOT per-phi optimality — read each row as "capture at phi
    # under the fixed config", not "best config at phi".
    row["locked_scope"] = "min_hold_bars + side_threshold fixed across phi"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phis", default="-0.3,0,0.1,0.2,0.35,0.5,0.7",
                    help="comma-separated phi values to sweep")
    ap.add_argument("--config", default=str(ROOT / "configs" / "exp_ar.yaml"))
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds (overrides --seed; one run per seed)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    phis = [float(x) for x in args.phis.split(",") if x.strip()]
    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()]
             if args.seeds else [int(args.seed)])
    template = Path(args.config)
    if not template.exists():
        raise SystemExit(f"template config not found: {template}")
    template_cfg = yaml.safe_load(template.read_text())
    ar_noise = float((template_cfg.get("data") or {}).get("ar_noise", 0.0))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = SWEEP / f"ar_sweep_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "ar_sweep.csv"

    jobs = [(phi, seed) for phi in phis for seed in seeds]
    print(
        f"sweep: operating config LOCKED from {template.name} "
        f"(min_hold_bars + side_threshold fixed, ar_noise={ar_noise:.2f}). "
        f"{len(phis)} phi × {len(seeds)} seed = {len(jobs)} runs. "
        f"Rows are 'capture at phi under fixed config', NOT per-phi optimal configs.",
        flush=True,
    )

    rows = []
    sweep_bar = tqdm(jobs, desc="AR sweep", unit="run", colour="cyan", leave=True)
    for phi, seed in sweep_bar:
        t1 = time.monotonic()
        sweep_bar.set_postfix_str(f"phi={phi:+.2f} seed={seed} measuring ar1…", refresh=True)
        rho1, per_std = _measured_ar1(phi, seed, ar_noise)
        sweep_bar.set_postfix_str(
            f"phi={phi:+.2f} seed={seed} ar1={rho1:+.3f} training…", refresh=True,
        )
        row = _run_one(phi, args.timesteps, template, run_dir, seed)
        row["measured_ar1"] = rho1
        row["per_step_std"] = per_std
        rows.append(row)
        elapsed = time.monotonic() - t1
        remaining = len(jobs) - len(rows)
        eta = elapsed * remaining if remaining > 0 else 0.0
        sweep_bar.set_postfix_str(
            f"phi={phi:+.2f} seed={seed} ret={row['total_return']:+.3f} "
            f"sh={row['sharpe']:+.1f} ~{eta:.0f}s left",
            refresh=True,
        )
        tqdm.write(
            f"phi={phi:+.2f} seed={seed} ar1={rho1:+.3f} ret={row['total_return']:+.4f} "
            f"sharpe={row['sharpe']:+.2f} trades={row['trades']} "
            f"({elapsed:.0f}s this run, ~{eta:.0f}s remaining)",
        )
    sweep_bar.close()

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"sweep done: {csv_path}")
    print(f"{'phi':>6} {'seed':>5} {'ar1':>7} {'ret':>9} {'sharpe':>8} {'sortino':>8} {'mdd':>9} {'ulcer':>8} {'upi':>9} {'trades':>7} {'win%':>6} {'gross':>9}")
    for r in rows:
        ui = "" if r["ulcer_index"] != r["ulcer_index"] else f"{r['ulcer_index']:8.4f}"
        upi = "" if r["upi"] != r["upi"] else f"{r['upi']:9.2f}"
        print(f"{r['phi']:+.3f} {r['seed']:5d} {r['measured_ar1']:+.3f} {r['total_return']:+.4f} "
              f"{r['sharpe']:8.2f} {r['sortino']:8.2f} {r['max_drawdown']:9.4f} "
              f"{ui} {upi} {r['trades']:7d} {100 * r['winrate']:5.1f}% {r['gross_pnl']:+9.1f}")
    if len(seeds) > 1 and len(phis) == 1:
        sh = [r["sharpe"] for r in rows]
        ret = [r["total_return"] for r in rows]
        print(f"seed summary: n={len(rows)}  sharpe mean={np.nanmean(sh):+.2f} "
              f"std={np.nanstd(sh):.2f}  ret mean={np.nanmean(ret):+.4f} "
              f"std={np.nanstd(ret):.4f}")

    if not args.no_plot:
        _plot(csv_path, run_dir / "ar_sweep.png")


def _fvals(rows, key):
    return [float("nan") if not r.get(key) or r.get(key) in ("nan", "") else float(r[key]) for r in rows]


def _plot(csv_path: Path, out_png: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = list(csv.DictReader(open(csv_path)))
    phis = [float(r["phi"]) for r in rows]
    seeds = [int(r["seed"]) if r.get("seed") not in (None, "") else 0 for r in rows]
    vs_seed = len(set(phis)) == 1 and len(set(seeds)) > 1
    xs = seeds if vs_seed else phis
    xlab = "seed" if vs_seed else "\u03c6"
    ar1 = [float(r["measured_ar1"]) for r in rows]
    ret = [float(r["total_return"]) for r in rows]
    sharpe = [float(r["sharpe"]) for r in rows]
    mdd = [float(r["max_drawdown"]) for r in rows]
    trades = [int(r["trades"]) for r in rows]
    ulcer = _fvals(rows, "ulcer_index")
    upi = _fvals(rows, "upi")

    fig = make_subplots(rows=3, cols=2, subplot_titles=(
        f"Total return vs {xlab}", f"Sharpe vs {xlab}",
        f"Ulcer Index vs {xlab}", f"UPI vs {xlab}",
        f"Max drawdown vs {xlab}", f"Trades vs {xlab}"))
    fig.add_trace(go.Scatter(x=xs, y=ret, mode="lines+markers", name="total_return"), 1, 1)
    fig.add_hline(y=0, line=dict(color="#FF4D6D", width=1, dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=sharpe, mode="lines+markers", name="sharpe"), 1, 2)
    fig.add_hline(y=0, line=dict(color="#FF4D6D", width=1, dash="dash"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=ulcer, mode="lines+markers", name="ulcer_index"), 2, 1)
    fig.add_hline(y=1, line=dict(color="#FF4D6D", width=1, dash="dash"), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=upi, mode="lines+markers", name="upi"), 2, 2)
    fig.add_hline(y=0, line=dict(color="#FF4D6D", width=1, dash="dash"), row=2, col=2)
    fig.add_trace(go.Scatter(x=xs, y=mdd, mode="lines+markers", name="max_drawdown"), 3, 1)
    fig.add_trace(go.Scatter(x=xs, y=trades, mode="lines+markers", name="trades"), 3, 2)
    fig.update_xaxes(title_text=xlab)
    fig.update_layout(template="plotly_dark", height=1080, showlegend=False)
    fig.write_image(str(out_png))
    print(f"sweep plot -> {out_png}")


if __name__ == "__main__":
    main()