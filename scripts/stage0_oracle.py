"""Stage 0 item 1: oracle φ-trader on the locked walk-forward harness.

Proves H1 — planted AR alpha is economically recoverable under the same
splits, fees, and gates used for HRL — without any learning.

    caffeinate -dims venv/bin/python scripts/stage0_oracle.py \
      --config configs/stage0.yaml \
      --gates roadmap/stage0_gates.yaml

Oracle rule (causal):
  phi > 0  → momentum: side = sign(ret1)
  phi < 0  → mean-revert: side = -sign(ret1)
  |ret1| <= deadzone → flat

Uses the worker continuous action in [-1, 1] with the Stage 0 env costs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agents import build_bundle, make_env  # noqa: E402
from config import load  # noqa: E402
from data import FEATURES, slice_bundle  # noqa: E402
from splits import assert_no_leakage, walk_forward_splits  # noqa: E402
from stage0 import _fold_cfg, _load_gates, _metric_block, _setup_logging  # noqa: E402

RET1_IDX = FEATURES.index("ret1")


def _oracle_action(obs: mx.array, phi: float, deadzone: float) -> mx.array:
    """Map current ret1 → continuous side in [-1, 1]."""
    ret1 = obs[:, RET1_IDX]
    side = mx.sign(ret1)
    if float(phi) < 0.0:
        side = -side
    elif float(phi) == 0.0:
        side = mx.zeros_like(side)
    if deadzone > 0.0:
        side = mx.where(mx.abs(ret1) > deadzone, side, mx.zeros_like(side))
    return side[:, None].astype(mx.float32)


def _equity(env) -> mx.array:
    """Per-env mark-to-market equity (isolated / cross aware)."""
    st = env._state
    bal, q, entry, coll = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    t = mx.minimum(env._steps, env.T - 1)
    px = mx.take(env.closes_flat, env.sym_off + t)
    if env.margin_mode == "cross":
        return bal + q * (px - entry)
    return bal + coll + q * (px - entry)


def run_oracle_segment(cfg: dict, bundle, phi: float, deadzone: float) -> dict:
    """Roll one bundle segment with the oracle; return Stage-0-shaped metrics."""
    cfg_e = dict(cfg)
    env_cfg = dict(cfg_e.get("env") or {})
    # One env per symbol is enough: oracle is deterministic given prices.
    env_cfg["n_envs_per_symbol"] = 1
    cfg_e["env"] = env_cfg

    env = make_env(cfg_e, "continuous", goal_dim=0, bundle=bundle, trade_knob=1.0)
    obs, _ = env.reset()
    net = [float(mx.mean(_equity(env)).item())]
    for _ in range(env.T - 1):
        act = _oracle_action(obs, phi=phi, deadzone=deadzone)
        obs, _, done, _ = env.step(act)
        net.append(float(mx.mean(_equity(env)).item()))
        # Auto-reset handled inside env via prev_done mask.
        del done
    return _metric_block(cfg, {"net": np.asarray(net, dtype=float)})


def run_stage0_oracle(
    cfg: dict,
    gates_doc: dict,
    run_dir: Path,
    log: logging.Logger,
    deadzone: float = 0.0,
) -> dict:
    from dirty_fin_reports.simple.gates import format_gate_report, score_stage0_gates

    scfg = _fold_cfg(cfg)
    n_folds = int(scfg.get("n_folds", 3))
    train_bars = int(scfg.get("train_bars", 300))
    test_bars = int(scfg.get("test_bars", 100))
    purge_bars = int(scfg.get("purge_bars", 20))
    embargo_bars = int(scfg.get("embargo_bars", 20))
    mode = str(scfg.get("mode", "expanding"))
    seeds = [int(s) for s in scfg.get("seeds", [cfg.get("seed", 42)])]
    n_steps = int((cfg.get("data") or {}).get("n_steps", 400))
    phi = float((cfg.get("data") or {}).get("ar", 0.0))
    ar_noise = float((cfg.get("data") or {}).get("ar_noise", 0.0))

    folds = walk_forward_splits(
        n_steps,
        n_folds=n_folds,
        train_bars=train_bars,
        test_bars=test_bars,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        mode=mode,
    )
    assert_no_leakage(folds)
    log.info(
        "oracle: phi=%.4f ar_noise=%.4f deadzone=%.6f folds=%d seeds=%s",
        phi, ar_noise, deadzone, n_folds, seeds,
    )

    out = run_dir / "stage0_oracle"
    out.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []
    jobs = [(seed, fold) for seed in seeds for fold in folds]
    for seed, fold in tqdm(jobs, desc="oracle folds", unit="fold"):
        fold_dir = out / f"seed{seed}" / f"fold{fold.fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        cfg_seed = dict(cfg)
        cfg_seed["seed"] = int(seed)
        full = build_bundle(cfg_seed)
        train_bundle = slice_bundle(full, fold.train_start, fold.train_end)
        test_bundle = slice_bundle(full, fold.test_start, fold.test_end)
        is_metrics = run_oracle_segment(cfg_seed, train_bundle, phi, deadzone)
        oos_metrics = run_oracle_segment(cfg_seed, test_bundle, phi, deadzone)
        row = {
            "seed": int(seed),
            "fold": int(fold.fold),
            "split": fold.to_dict(),
            "system": "oracle_ret1",
            "phi": phi,
            "ar_noise": ar_noise,
            "is": is_metrics,
            "oos": oos_metrics,
        }
        fold_rows.append(row)
        with open(fold_dir / "fold_metrics.json", "w") as fh:
            json.dump(row, fh, indent=2, default=str)
        log.info(
            "oracle seed=%d fold=%d IS ret=%s UPI=%s | OOS ret=%s UPI=%s",
            seed, fold.fold,
            is_metrics.get("total_return"), is_metrics.get("upi"),
            oos_metrics.get("total_return"), oos_metrics.get("upi"),
        )

    scored = score_stage0_gates(fold_rows, gates_doc.get("gates") or {})
    report = {
        "system": "oracle_ret1",
        "phi": phi,
        "ar_noise": ar_noise,
        "deadzone": deadzone,
        "gates_frozen": True,
        "protocol_version": gates_doc.get("protocol_version"),
        "folds": fold_rows,
        "gate_report": scored,
    }
    with open(out / "stage0_oracle_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    text = format_gate_report(scored)
    (out / "stage0_oracle_gates.txt").write_text(text + "\n")
    print(text)
    print(f"artifacts -> {out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0 oracle φ-trader (H1)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "stage0.yaml"))
    ap.add_argument("--gates", default=str(ROOT / "roadmap" / "stage0_gates.yaml"))
    ap.add_argument("--ar", type=float, default=None, help="override data.ar (phi)")
    ap.add_argument("--ar-noise", type=float, default=None, help="override data.ar_noise")
    ap.add_argument(
        "--deadzone",
        type=float,
        default=0.0,
        help="|ret1| below this → flat (default 0)",
    )
    ap.add_argument("--tag", default="oracle", help="log dir tag suffix")
    args = ap.parse_args()
    cfg = load(args.config)
    cfg.setdefault("data", {})
    if args.ar is not None:
        cfg["data"]["ar"] = float(args.ar)
    if args.ar_noise is not None:
        cfg["data"]["ar_noise"] = float(args.ar_noise)
    gates_doc = _load_gates(Path(args.gates))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    phi = float(cfg["data"].get("ar", 0.0))
    kappa = float(cfg["data"].get("ar_noise", 0.0))
    run_dir = ROOT / "logs" / f"{ts}-{os.getpid()}-stage0-{args.tag}-phi{phi:g}-k{kappa:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    report = run_stage0_oracle(cfg, gates_doc, run_dir, log, deadzone=float(args.deadzone))
    overall = bool((report.get("gate_report") or {}).get("overall_pass"))
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
