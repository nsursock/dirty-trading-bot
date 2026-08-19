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

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "logs" / "sweeps"

_DONE_RE = re.compile(
    r"-> (?P<out>\S+)  \{\'sharpe\': np\.float64\((?P<sharpe>[-\d.eE+]+)\), "
    r"\'sortino\': (?:np\.float64\((?P<sortino>[-\d.eE+]+)\)|None), "
    r"\'max_drawdown\': (?P<mdd>[-\d.eE+]+), "
    r"\'cagr\': np\.float64\((?P<cagr>[-\d.eE+]+)\), "
    r"\'final_equity\': (?P<eq>[-\d.eE+]+), "
    r"\'total_return\': (?P<ret>[-\d.eE+]+), \'return_basis\': \'(?P<basis>\w+)\'.*\}$"
)


def _measured_ar1(phi: float, seed: int) -> tuple[float, float]:
    """Lag-1 autocorr + per-step return std from the same bundle path as training."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from data import generate

    b = generate(
        symbols={"BTC": (0.30, 0.60, 60_000.0)},
        n_steps=20000, seed=seed, low_tf=5, high_tf=5, regime="neutral",
        ar_coef=phi,
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


def _run_one(phi: float, timesteps: int, template: Path, out: Path, seed: int) -> dict:
    cfg = yaml.safe_load(template.read_text())
    cfg["data"]["ar"] = float(phi)
    cfg["eval"]["deterministic"] = True
    cfg_path = out / f"phi_{phi:+.2f}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [sys.executable, "scripts/main.py", "full", "--config", str(cfg_path),
         "--timesteps", str(timesteps)],
        cwd=ROOT, capture_output=True, text=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    m = _DONE_RE.search(stdout)
    if m is None:
        raise RuntimeError(
            f"phi={phi}: no metrics line in output\n{stdout[-1500:]}\n{stderr[-1500:]}"
        )
    g = m.groupdict()
    ts = _trade_stats(Path(g["out"]))
    row = {
        "phi": float(phi),
        "total_return": float(g["ret"]),
        "sharpe": float(g["sharpe"]),
        "sortino": float(g["sortino"]) if g["sortino"] else float("nan"),
        "max_drawdown": float(g["mdd"]),
        "final_equity": float(g["eq"]),
        "cagr": float(g["cagr"]),
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
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    phis = [float(x) for x in args.phis.split(",") if x.strip()]
    template = Path(args.config)
    if not template.exists():
        raise SystemExit(f"template config not found: {template}")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = SWEEP / f"ar_sweep_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "ar_sweep.csv"

    print(
        f"sweep: phi varies, operating config LOCKED from {template.name} "
        f"(min_hold_bars + side_threshold fixed). Rows are 'capture at phi under "
        f"fixed config', NOT per-phi optimal configs.",
        flush=True,
    )

    rows = []
    t0 = time.monotonic()
    for phi in phis:
        t1 = time.monotonic()
        rho1, per_std = _measured_ar1(phi, args.seed)
        row = _run_one(phi, args.timesteps, template, run_dir, args.seed)
        row["measured_ar1"] = rho1
        row["per_step_std"] = per_std
        rows.append(row)
        elapsed = time.monotonic() - t1
        remaining = len(phis) - len(rows)
        eta = elapsed * remaining if remaining > 0 else 0.0
        print(
            f"phi={phi:+.2f} ar1={rho1:+.3f} ret={row['total_return']:+.4f} "
            f"sharpe={row['sharpe']:+.2f} trades={row['trades']} "
            f"({elapsed:.0f}s this run, ~{eta:.0f}s remaining)",
            flush=True,
        )

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"sweep done: {csv_path}")
    print(f"{'phi':>6} {'ar1':>7} {'ret':>9} {'sharpe':>8} {'sortino':>8} {'mdd':>9} {'trades':>7} {'win%':>6} {'gross':>9}")
    for r in rows:
        print(f"{r['phi']:+.3f} {r['measured_ar1']:+.3f} {r['total_return']:+.4f} "
              f"{r['sharpe']:8.2f} {r['sortino']:8.2f} {r['max_drawdown']:9.4f} "
              f"{r['trades']:7d} {100 * r['winrate']:5.1f}% {r['gross_pnl']:+9.1f}")

    if not args.no_plot:
        _plot(csv_path, run_dir / "ar_sweep.png")


def _plot(csv_path: Path, out_png: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = list(csv.DictReader(open(csv_path)))
    phis = [float(r["phi"]) for r in rows]
    ar1 = [float(r["measured_ar1"]) for r in rows]
    ret = [float(r["total_return"]) for r in rows]
    sharpe = [float(r["sharpe"]) for r in rows]
    mdd = [float(r["max_drawdown"]) for r in rows]
    trades = [int(r["trades"]) for r in rows]

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Total return vs \u03c6", "Sharpe vs \u03c6",
        "Max drawdown vs \u03c6", "Trades vs \u03c6"))
    fig.add_trace(go.Scatter(x=phis, y=ret, mode="lines+markers", name="total_return"), 1, 1)
    fig.add_hline(y=0, line=dict(color="#FF4D6D", width=1, dash="dash"), row=1, col=1)
    fig.add_trace(go.Scatter(x=phis, y=sharpe, mode="lines+markers", name="sharpe"), 1, 2)
    fig.add_hline(y=0, line=dict(color="#FF4D6D", width=1, dash="dash"), row=1, col=2)
    fig.add_trace(go.Scatter(x=phis, y=mdd, mode="lines+markers", name="max_drawdown"), 2, 1)
    fig.add_trace(go.Scatter(x=phis, y=trades, mode="lines+markers", name="trades"), 2, 2)
    fig.update_xaxes(title_text="\u03c6")
    fig.update_layout(template="plotly_dark", height=720, showlegend=False)
    fig.write_image(str(out_png))
    print(f"sweep plot -> {out_png}")


if __name__ == "__main__":
    main()