"""Stage 0 runner: walk-forward GBM+AR through locked validation gates.

    caffeinate -dims venv/bin/python scripts/stage0.py \
      --config configs/stage0_smoke.yaml \
      --gates roadmap/stage0_gates.yaml

Trains HRL on each causal train window, scores IS + OOS segments, then asks
``dirty_fin_reports.simple.gates`` to evaluate the frozen thresholds.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx  # noqa: F401
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agents import JointHRL, build_bundle  # noqa: E402
from config import load  # noqa: E402
from data import slice_bundle  # noqa: E402
from report import run_test  # noqa: E402
from splits import assert_no_leakage, walk_forward_splits  # noqa: E402


def _run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    d = ROOT / "logs" / f"{ts}-{os.getpid()}-stage0"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("trading")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    fh = logging.FileHandler(run_dir / "run.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(fh)
    return logger


def _load_gates(path: Path) -> dict:
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or "gates" not in doc:
        raise SystemExit(f"gates file missing 'gates': {path}")
    if not doc.get("frozen", False):
        raise SystemExit(f"refusing unfrozen gates file: {path}")
    return doc


def _fold_cfg(cfg: dict) -> dict:
    return dict(cfg.get("stage0") or cfg.get("cv") or {})


def _metric_block(cfg: dict, raw: dict) -> dict:
    from dirty_fin_reports.simple.metrics import metrics, periods_per_year

    d = cfg.get("data") or {}
    r = cfg.get("returns") or {}
    tf = str((d.get("timeframes") or {}).get("low", "5m"))
    ppy = periods_per_year(tf)
    net = np.asarray(raw["net"], dtype=float)
    m = metrics(
        net,
        periods_per_year=ppy,
        freq=str(r.get("freq", "daily")),
        rf_annual=float(r.get("rf_annual", 0.045)),
    )
    return {
        "total_return": m.get("total_return"),
        "upi": m.get("upi"),
        "sharpe": m.get("sharpe"),
        "sortino": m.get("sortino"),
        "cagr": m.get("cagr"),
        "max_drawdown": m.get("max_drawdown"),
        "ulcer_index": m.get("ulcer_index"),
        "final_equity": m.get("final_equity"),
    }


def _free(j: JointHRL | None) -> None:
    if j is None:
        return
    j.worker_env = None
    j.mgr_env = None
    if hasattr(j, "worker") and j.worker is not None:
        j.worker.replay_buffer = None
    j.manager = None
    j.worker = None
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def _eval_segment(cfg, j: JointHRL, bundle, seed_offset: int) -> dict:
    raw = run_test(
        cfg,
        j.manager,
        j.worker,
        norm_state=j.worker_env.norm_state,
        seed_offset=seed_offset,
        deterministic=bool((cfg.get("eval") or {}).get("deterministic", True)),
        bundle=bundle,
    )
    return _metric_block(cfg, raw)


def _save_fold(j: JointHRL, train_dir: Path, config_path: Path | None = None) -> Path:
    """Persist checkpoint + rename progress CSVs to match ``main.py`` artifacts."""
    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    j.save(train_dir, config_path=str(config_path) if config_path is not None else None)
    for src, dst in [("ppo_progress.csv", "manager_ppo.csv"), ("sac_progress.csv", "worker_sac.csv")]:
        s, d = train_dir / src, train_dir / dst
        if s.exists():
            s.rename(d)
    return train_dir


def run_stage0(
    cfg: dict,
    gates_doc: dict,
    run_dir: Path,
    log: logging.Logger,
    config_path: Path | None = None,
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
        "stage0: method=walk_forward mode=%s folds=%d seeds=%s purge=%d embargo=%d",
        mode, n_folds, seeds, purge_bars, embargo_bars,
    )
    for f in folds:
        log.info("stage0 fold %d: %s", f.fold, f.to_dict())

    out = run_dir / "stage0"
    out.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []

    jobs = [(seed, fold) for seed in seeds for fold in folds]
    for seed, fold in tqdm(jobs, desc="stage0 folds", unit="fold"):
        fold_dir = out / f"seed{seed}" / f"fold{fold.fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        cfg_seed = dict(cfg)
        cfg_seed["seed"] = int(seed)
        full = build_bundle(cfg_seed)
        train_bundle = slice_bundle(full, fold.train_start, fold.train_end)
        test_bundle = slice_bundle(full, fold.test_start, fold.test_end)
        log.info(
            "stage0 seed=%d fold=%d train=[%d,%d) test=[%d,%d)",
            seed, fold.fold, fold.train_start, fold.train_end,
            fold.test_start, fold.test_end,
        )
        j = None
        try:
            j = JointHRL(
                cfg_seed,
                log_dir=str(fold_dir / "training"),
                config_path=str(config_path) if config_path is not None else None,
                bundle=train_bundle,
            )
            j.learn(
                log_interval=cfg_seed.get("train", {}).get("log_interval", 1),
                log_every=cfg_seed.get("train", {}).get("log_every", 0),
            )
            _save_fold(j, fold_dir / "training", config_path=config_path)
            is_metrics = _eval_segment(cfg_seed, j, train_bundle, seed_offset=0)
            oos_metrics = _eval_segment(cfg_seed, j, test_bundle, seed_offset=0)
        finally:
            _free(j)

        row = {
            "seed": int(seed),
            "fold": int(fold.fold),
            "split": fold.to_dict(),
            "is": is_metrics,
            "oos": oos_metrics,
        }
        fold_rows.append(row)
        with open(fold_dir / "fold_metrics.json", "w") as fh:
            json.dump(row, fh, indent=2, default=str)
        log.info(
            "stage0 seed=%d fold=%d IS ret=%s UPI=%s | OOS ret=%s UPI=%s",
            seed, fold.fold,
            is_metrics.get("total_return"), is_metrics.get("upi"),
            oos_metrics.get("total_return"), oos_metrics.get("upi"),
        )

    scored = score_stage0_gates(fold_rows, gates_doc.get("gates") or {})
    report = {
        "stage": 0,
        "protocol_version": gates_doc.get("protocol_version"),
        "gates_frozen": bool(gates_doc.get("frozen")),
        "method": gates_doc.get("method"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_steps": n_steps,
            "n_folds": n_folds,
            "train_bars": train_bars,
            "test_bars": test_bars,
            "purge_bars": purge_bars,
            "embargo_bars": embargo_bars,
            "mode": mode,
            "seeds": seeds,
            "ar": (cfg.get("data") or {}).get("ar"),
            "ar_noise": (cfg.get("data") or {}).get("ar_noise"),
            "regime": (cfg.get("data") or {}).get("regime"),
        },
        "folds": fold_rows,
        "gate_report": scored,
        "overall_pass": bool(scored.get("overall_pass")),
    }
    with open(out / "stage0_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    text = format_gate_report(scored)
    (out / "stage0_gates.txt").write_text(text)
    log.info("stage0 overall_pass=%s\n%s", report["overall_pass"], text)
    print(text)
    print(f"overall_pass={report['overall_pass']}")
    print(f"artifacts -> {out}")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0: validate the validator on GBM+AR")
    ap.add_argument("--config", default=str(ROOT / "configs" / "stage0_smoke.yaml"))
    ap.add_argument("--gates", default=str(ROOT / "roadmap" / "stage0_gates.yaml"))
    ap.add_argument("--ar", type=float, default=None, help="override data.ar (phi)")
    ap.add_argument("--ar-noise", type=float, default=None, help="override data.ar_noise")
    ap.add_argument("--tag", default="stage0", help="extra log-dir tag")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    gates_path = Path(args.gates)
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    if not gates_path.exists():
        raise SystemExit(f"gates not found: {gates_path}")

    cfg = load(cfg_path)
    cfg.setdefault("data", {})
    if args.ar is not None:
        cfg["data"]["ar"] = float(args.ar)
    if args.ar_noise is not None:
        cfg["data"]["ar_noise"] = float(args.ar_noise)
    phi = float(cfg["data"].get("ar", 0.0))
    kappa = float(cfg["data"].get("ar_noise", 0.0))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = ROOT / "logs" / f"{ts}-{os.getpid()}-{args.tag}-phi{phi:g}-k{kappa:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    shutil.copy2(cfg_path, run_dir / cfg_path.name)
    shutil.copy2(gates_path, run_dir / gates_path.name)
    # Persist effective overrides for the record.
    def _plain(obj):
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_plain(v) for v in obj]
        return obj

    with open(run_dir / "effective_data.yaml", "w") as fh:
        yaml.safe_dump({"data": _plain(cfg.get("data") or {})}, fh, sort_keys=False)
    gates_doc = _load_gates(gates_path)
    log.info(
        "stage0 start config=%s gates=%s phi=%s ar_noise=%s",
        cfg_path, gates_path, phi, kappa,
    )
    report = run_stage0(cfg, gates_doc, run_dir, log, config_path=cfg_path)
    return 0 if report.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
