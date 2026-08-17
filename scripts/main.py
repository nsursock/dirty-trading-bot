"""Entry point — three execution modes.

    python main.py train  [--config configs/normal.yaml]
    python main.py test   [--checkpoint logs/<ts>/training]
    python main.py full   [--config configs/normal.yaml]

Writes a timestamped run folder ``logs/<timestamp>/`` with ``run.log`` at its
root (WARN/ERROR only) and ``training/`` / ``testing/`` artifacts beneath.
Screen output is limited to tqdm progress bars.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import mlx.core as mx  # noqa: F401  (ensure mlx import before dirty_mlx_ml)

from agents import JointHRL
from config import load, load_smoke
from report import generate_report, ml_health

ROOT = Path(__file__).resolve().parents[1]


def _run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    d = ROOT / "logs" / ts
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


def _load_cfg(args) -> dict:
    return load(args.config) if args.config else load_smoke()


def _save(j: JointHRL, train_dir: Path) -> Path:
    train_dir.mkdir(parents=True, exist_ok=True)
    j.save(train_dir)
    for src, dst in [("ppo_progress.csv", "manager_ppo.csv"), ("sac_progress.csv", "worker_sac.csv")]:
        s, d = train_dir / src, train_dir / dst
        if s.exists():
            s.rename(d)
    return train_dir


def _train(cfg, run_dir: Path, log) -> JointHRL:
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    log.info("training start: total_timesteps=%s", cfg.get("train", {}).get("total_timesteps"))
    j = None
    try:
        j = JointHRL(cfg, log_dir=str(train_dir))
        j.learn(log_interval=cfg.get("train", {}).get("log_interval", 1),
                log_every=cfg.get("train", {}).get("log_every", 0))
    except KeyboardInterrupt:
        log.warning("training interrupted by user; saving checkpoint")
        if j is not None:
            j.save(train_dir)
        raise
    except Exception as e:
        log.exception("training failed: %s", e)
        raise
    _save(j, train_dir)
    theme = (cfg.get("report") or {}).get("theme", "synthwave")
    ml_health(train_dir / "manager_ppo.csv", train_dir / "manager_diag.png", "manager", theme=theme)
    ml_health(train_dir / "worker_sac.csv", train_dir / "worker_diag.png", "worker", theme=theme)
    log.info("training done: artifacts -> %s", train_dir)
    return j


def _latest_checkpoint() -> Path | None:
    dirs = sorted((ROOT / "logs").glob("*/training"), key=lambda p: p.parent.name)
    return dirs[-1] if dirs else None


def _test(cfg, run_dir: Path, j: JointHRL) -> dict:
    return generate_report(cfg, j.manager, j.worker, run_dir / "testing",
                           norm_state=j.worker_env.norm_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test", "full"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    run_dir = _run_dir()
    log = _setup_logging(run_dir)
    cfg = _load_cfg(args)
    log.info("run started: mode=%s run_dir=%s", args.mode, run_dir)
    log.debug("config: %s", dict(cfg) if hasattr(cfg, "items") else cfg)

    if args.mode == "train":
        _train(cfg, run_dir, log)
        log.info("train complete: artifacts -> %s", run_dir / "training")
        print(f"trained -> {run_dir / 'training'}")
    elif args.mode == "full":
        j = _train(cfg, run_dir, log)
        out = _test(cfg, run_dir, j)
        log.info("full run complete: %s", out["metrics"])
        print(f"done -> {out['out_dir']}  {out['metrics']}")
    else:
        ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
        if ckpt is None or not (ckpt / "manager_policy.safetensors").exists():
            log.error("no checkpoint found; run `train` first")
            raise SystemExit("no checkpoint found; run `train` first")
        log.info("loading checkpoint: %s", ckpt)
        j = JointHRL(cfg)
        j.load(ckpt)
        out = _test(cfg, run_dir, j)
        log.info("test complete: %s", out["metrics"])
        print(f"done -> {out['out_dir']}  {out['metrics']}")


if __name__ == "__main__":
    main()
