"""Entry point — three execution modes.

    python main.py train  [--config configs/normal.yaml]
    python main.py test   [--checkpoint logs/<ts>/training] [--config cfg.yaml] [--force]
    python main.py full   [--config configs/normal.yaml]

Writes a timestamped run folder ``logs/<timestamp>/`` with a copy of the
source YAML (original filename), ``run.log`` at its root, and ``training/`` /
``testing/`` artifacts beneath.
Screen output is limited to tqdm progress bars.

Test-mode config binding: every checkpoint saved by ``JointHRL.save`` carries
``manifest.json`` (config hash, dims, seed, git SHAs). ``test`` auto-loads the
config the checkpoint was trained with and refuses to score it under a
different config unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import mlx.core as mx  # noqa: F401  (ensure mlx import before dirty_mlx_ml)

from agents import JointHRL, config_hash
from config import load
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


def _config_path(args) -> Path:
    return Path(args.config) if args.config else ROOT / "configs" / "smoke.yaml"


def _copy_config(src: Path, run_dir: Path) -> Path:
    dst = run_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _save(j: JointHRL, train_dir: Path, config_path: Path | None = None) -> Path:
    train_dir.mkdir(parents=True, exist_ok=True)
    j.save(train_dir, config_path=config_path)
    for src, dst in [("ppo_progress.csv", "manager_ppo.csv"), ("sac_progress.csv", "worker_sac.csv")]:
        s, d = train_dir / src, train_dir / dst
        if s.exists():
            s.rename(d)
    return train_dir


def _train(cfg, run_dir: Path, log, config_path: Path) -> JointHRL:
    train_dir = run_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    log.info("training start: total_timesteps=%s", cfg.get("train", {}).get("total_timesteps"))
    config_path = config_path.resolve()
    j = None
    try:
        j = JointHRL(cfg, log_dir=str(train_dir), config_path=str(config_path))
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
    _save(j, train_dir, config_path=config_path)
    theme = (cfg.get("report") or {}).get("theme", "synthwave")
    ml_health(train_dir / "manager_ppo.csv", train_dir / "manager_diag.png", "manager", theme=theme)
    ml_health(train_dir / "worker_sac.csv", train_dir / "worker_diag.png", "worker", theme=theme)
    log.info("training done: artifacts -> %s", train_dir)
    return j


def _latest_checkpoint() -> Path | None:
    dirs = sorted((ROOT / "logs").glob("*/training"), key=lambda p: p.parent.name)
    return dirs[-1] if dirs else None


def _load_manifest(ckpt: Path) -> dict | None:
    p = ckpt / "manifest.json"
    if not p.exists():
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        log = logging.getLogger("trading")
        log.warning("manifest %s unreadable (%s); proceeding unchecked", p, e)
        return None


def _resolve_checkpoint(args) -> Path:
    ck = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint()
    if ck is None or not (ck / "manager_policy.safetensors").exists():
        log = logging.getLogger("trading")
        log.error("no checkpoint found; run `train` first")
        raise SystemExit("no checkpoint found; run `train` first")
    return ck


def _test_config_source(args, ckpt: Path, log) -> tuple[Path, dict | None]:
    """Resolve which config file to score a checkpoint under.

    Auto-loads the manifest-referenced config when ``--config`` is absent;
    otherwise uses ``--config`` (or the smoke default). Refuses to score a
    checkpoint under a config that does not match its manifest hash unless
    ``--force`` is set.
    """
    manifest = _load_manifest(ckpt)
    explicit = args.config is not None
    if explicit:
        src = Path(args.config)
    elif manifest is not None and manifest.get("config_path"):
        p = Path(manifest["config_path"])
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            log.error(
                "manifest references config %s which no longer exists; pass --config or --force",
                p,
            )
            raise SystemExit(f"checkpoint config {p} is missing; pass --config or --force")
        src = p
    else:
        src = _config_path(args)

    if not src.exists():
        log.error("config %s does not exist", src)
        raise SystemExit(f"config {src} does not exist")

    cfg = load(src)
    if manifest is not None and not args.force:
        want = manifest.get("config_hash")
        got = config_hash(cfg)
        if want and want != got:
            log.error(
                "config mismatch: checkpoint is bound to config hash %s but %s hashes %s; "
                "pass --force to override",
                want, src, got,
            )
            raise SystemExit(f"config mismatch for checkpoint {ckpt}; pass --force to override")
    return src, manifest


def _assert_dims_against_manifest(j: JointHRL, ckpt: Path, manifest: dict | None, log):
    if manifest is None:
        log.warning("no manifest for %s; checkpoint was saved before manifests (config hash unverified)", ckpt)
        return
    dims = {
        "manager_obs_dim": j.obs_mgr_dim,
        "worker_obs_dim": j.worker_env.observation_space.shape[0],
        "goal_dim": j.goal_dim,
    }
    for key, want in dims.items():
        got = manifest.get(key)
        if got is not None and got != want:
            raise SystemExit(
                f"dimension mismatch on {key}: manifest={got} config={want}; "
                f"config and checkpoint were trained under different architectures"
            )


def _test(cfg, run_dir: Path, j: JointHRL) -> dict:
    return generate_report(cfg, j.manager, j.worker, run_dir / "testing",
                           norm_state=j.worker_env.norm_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "test", "full"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--force", action="store_true",
                    help="allow scoring a checkpoint under a config that disagrees with its manifest")
    args = ap.parse_args()

    run_dir = _run_dir()
    log = _setup_logging(run_dir)
    log.info("run started: mode=%s run_dir=%s", args.mode, run_dir)

    if args.mode == "test":
        ckpt = _resolve_checkpoint(args)
        src, manifest = _test_config_source(args, ckpt, log)
    else:
        ckpt, manifest = None, None
        src = _config_path(args)
    cfg = load(src)
    copied = _copy_config(src, run_dir)
    log.info("config: %s -> %s", src, copied)
    log.debug("config: %s", dict(cfg) if hasattr(cfg, "items") else cfg)

    if args.mode == "test":
        log.info("loading checkpoint: %s", ckpt)
        j = JointHRL(cfg)
        _assert_dims_against_manifest(j, ckpt, manifest, log)
        j.load(ckpt)
        out = _test(cfg, run_dir, j)
        log.info("test complete: %s", out["metrics"])
        print(f"done -> {out['out_dir']}  {out['metrics']}")
    elif args.mode == "train":
        _train(cfg, run_dir, log, config_path=src)
        log.info("train complete: artifacts -> %s", run_dir / "training")
        print(f"trained -> {run_dir / 'training'}")
    else:
        j = _train(cfg, run_dir, log, config_path=src)
        out = _test(cfg, run_dir, j)
        log.info("full run complete: %s", out["metrics"])
        print(f"done -> {out['out_dir']}  {out['metrics']}")


if __name__ == "__main__":
    main()