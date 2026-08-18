"""Optuna hyperparameter optimization with pruning (multi-tier tuning).

Tier 1 sweeps wide ranges on a short budget; tier 2 re-tunes around the tier-1
best with a longer budget. MedianPruner prunes mid-training via the joint
loop's per-cycle callback.

Search integrity: the objective scores every trial on the **validation**
bundle (``VALID_SEED_OFFSETS``), never the locked final-test seeds
(``TEST_SEED_OFFSETS``). After the search, the best trial is reported with a
Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) using the total trial
count. The locked test is reserved for ``main.py test`` on a trained
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import mlx.core as mx  # noqa: F401
import optuna

from agents import JointHRL
from config import load, load_smoke
from report import (
    TEST_SEED_OFFSETS,
    VALID_SEED_OFFSETS,
    _periods_per_year,
    dsr,
    validate,
)

log = logging.getLogger("trading")

ROOT = Path(__file__).resolve().parents[1]
MIN_VAL_SEEDS = 2

TIER1_SPEC = {
    "manager.learning_rate": (1e-4, 1e-2, True),
    "manager.gamma": (0.9, 0.999, False),
    "manager.ent_coef": (0.0, 0.1, False),
    "worker.learning_rate": (1e-4, 1e-2, True),
    "worker.tau": (0.001, 0.05, False),
    "hrl.goal_every": (2, 8, False),
}


def _get(cfg, path):
    node = cfg
    for k in path.split("."):
        node = node[k]
    return node


def _set(cfg, path, value):
    *parts, last = path.split(".")
    node = cfg
    for k in parts:
        node = node[k]
    node[last] = value


def _patch(cfg, trial, spec):
    out = copy.deepcopy(cfg)
    for path, (lo, hi, log) in spec.items():
        _set(out, path, trial.suggest_float(path, lo, hi, log=log))
    if "hrl.goal_every" in spec:
        _set(out, "hrl.goal_every", int(_get(out, "hrl.goal_every")))
    return out


def _narrow(cfg, best_params):
    spec = {}
    for path, (lo, hi, log) in TIER1_SPEC.items():
        b = best_params[path]
        r = max(hi - lo, 1e-6) * 0.5
        spec[path] = (max(lo, b - r), min(hi, b + r), log)
    return spec


def _objective(cfg, timesteps, spec, pruner_cb=True, n_val_seeds=MIN_VAL_SEEDS):
    def objective(trial):
        gc.collect()
        mx.metal.clear_cache()
        cfg2 = _patch(cfg, trial, spec)
        log.debug("optuna: sampled params=%s", {k: _get(cfg2, k) for k in spec})

        def on_iter(iteration, model):
            trial.report(model.last_ep_rew_mean, iteration)
            if pruner_cb and trial.should_prune():
                raise optuna.TrialPruned()

        j = JointHRL(cfg2)
        j.learn(total_timesteps=timesteps, log_interval=10_000_000, on_iter=on_iter)
        v = validate(cfg2, j.manager, j.worker, n_seeds=n_val_seeds,
                     norm_state=j.worker_env.norm_state)
        value = v["sharpe_mean"]
        trial.set_user_attr("val_sharpes", v["sharpe_list"])
        trial.set_user_attr("val_net", [float(x) for x in (v["nets"][0] if v["nets"] else [])])
        trial.set_user_attr("val_seed_offsets", v["seed_offsets"])
        log.info(
            "optuna: trial value=%.4f val_sharpe=%.4f+/-%.4f (offsets=%s) locked_test_offsets=%s",
            value, v["sharpe_mean"], v["sharpe_std"], v["seed_offsets"], list(TEST_SEED_OFFSETS),
        )
        return value

    return objective


def run_search(cfg, n_trials=20, tier1_steps=None, tier2_steps=None, n_val_seeds=MIN_VAL_SEEDS,
               out_dir=None):
    """Run the two-tier Optuna search on the validation bundle.

    The locked ``TEST_SEED_OFFSETS`` are never sampled by the objective.
    Returns ``(best_params, best_value, report)`` where ``report`` carries the
    best-trial DSR and the trial count.
    """
    if n_val_seeds < MIN_VAL_SEEDS:
        raise ValueError(f"need at least {MIN_VAL_SEEDS} validation seeds per trial")
    base_steps = cfg.get("train", {}).get("total_timesteps", 4096)
    tier1_steps = tier1_steps or base_steps // 8
    tier2_steps = tier2_steps or base_steps

    study1 = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    study1.optimize(
        _objective(cfg, tier1_steps, TIER1_SPEC, n_val_seeds=n_val_seeds),
        n_trials=max(1, n_trials // 2),
    )
    best_params = study1.best_params

    study2 = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    study2.optimize(
        _objective(cfg, tier2_steps, _narrow(cfg, best_params), n_val_seeds=n_val_seeds),
        n_trials=max(1, n_trials - n_trials // 2),
    )

    n_trials_total = len(study1.trials) + len(study2.trials)
    best_trial = study2.best_trial
    net = best_trial.user_attrs.get("val_net")
    ppy = _periods_per_year(cfg)
    if net and len(net) > 4:
        dsr_report = dsr(net, periods_per_year=ppy, n_trials=max(n_trials_total, 1))
    else:
        dsr_report = {"deflated_sharpe": 0.0, "dsr_probability": 0.0,
                      "expected_max_sharpe": 0.0, "sharpe": 0.0}
    dsr_report["n_trials"] = n_trials_total

    report = {
        "best_params": best_params,
        "best_validation_sharpe": float(best_trial.value),
        "best_validation_seed_offsets": best_trial.user_attrs.get("val_seed_offsets", []),
        "validation_seed_offsets": list(VALID_SEED_OFFSETS),
        "locked_test_seed_offsets": list(TEST_SEED_OFFSETS),
        "n_trials": n_trials_total,
        "dsr": dsr_report,
    }
    dump = dict(report)
    dump["dsr"] = {k: (float(v) if isinstance(v, float) else v) for k, v in dsr_report.items()}
    out_dir = out_dir or (ROOT / "logs" / f"optim_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "optim_result.json", "w") as fh:
        json.dump(dump, fh, indent=2)
    log.info("optuna: best validation sharpe=%.4f (%d trials) dsr_p=%.4f artifacts -> %s",
             best_trial.value, n_trials_total, dsr_report["dsr_probability"], out_dir)
    return best_params, best_trial.value, report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--tier1-steps", type=int, default=None)
    ap.add_argument("--tier2-steps", type=int, default=None)
    ap.add_argument("--val-seeds", type=int, default=MIN_VAL_SEEDS)
    args = ap.parse_args()
    cfg = load(args.config) if args.config else load_smoke()
    best, value, report = run_search(
        cfg, args.n_trials, args.tier1_steps, args.tier2_steps,
        n_val_seeds=args.val_seeds,
    )
    print(f"best validation sharpe={value:.4f} trials={report['n_trials']} "
          f"dsr_p={report['dsr']['dsr_probability']:.4f} params={best}")


if __name__ == "__main__":
    main()