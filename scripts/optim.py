"""Optuna hyperparameter optimization with pruning (multi-tier tuning).

Tier 1 sweeps wide ranges on a short budget; tier 2 re-tunes around the tier-1
best with a longer budget. MedianPruner prunes mid-training via the joint
loop's per-cycle callback.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import math

import mlx.core as mx  # noqa: F401
import optuna

from agents import JointHRL
from config import load, load_smoke
from report import metrics, run_test

log = logging.getLogger("trading")

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


def _objective(cfg, timesteps, spec, pruner_cb=True):
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
        m = metrics(run_test(cfg2, j.manager, j.worker)["net"])
        value = m.get("sharpe", m.get("sortino", m.get("total_return", -1e9)))
        log.info("optuna: trial value=%.4f metrics=%s", value, m)
        return value

    return objective


def run_search(cfg, n_trials=20, tier1_steps=None, tier2_steps=None):
    base_steps = cfg.get("train", {}).get("total_timesteps", 4096)
    tier1_steps = tier1_steps or base_steps // 8
    tier2_steps = tier2_steps or base_steps

    study1 = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    study1.optimize(_objective(cfg, tier1_steps, TIER1_SPEC), n_trials=max(1, n_trials // 2))
    best_params = study1.best_params

    study2 = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    study2.optimize(
        _objective(cfg, tier2_steps, _narrow(cfg, best_params)), n_trials=max(1, n_trials - n_trials // 2)
    )
    return study2.best_params, study2.best_value


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--tier1-steps", type=int, default=None)
    ap.add_argument("--tier2-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load(args.config) if args.config else load_smoke()
    best, value = run_search(cfg, args.n_trials, args.tier1_steps, args.tier2_steps)
    print(f"best value={value:.4f} params={best}")


if __name__ == "__main__":
    main()
