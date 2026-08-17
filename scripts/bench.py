"""Benchmark / instrumentation harness for the joint-HRL milestones (consensus.md).

Subcommands:
  correctness   joint loop N cycles @ cfg envs: per-cycle FPS / reward / memory + NaN scan
  env           env-only collection throughput (no gradient calls)
  joint         joint loop (rollout + PPO + SAC) steps/sec + peak memory over N cycles
  sweep         env-throughput vs num_envs (16..1024)
"""

from __future__ import annotations

import argparse
import logging
import time

import mlx.core as mx
from mlx.utils import tree_flatten

from agents import JointHRL, make_env
from config import load, load_smoke

log = logging.getLogger("trading")


def _mem():
    return mx.get_active_memory(), mx.get_peak_memory()


def correctness(cfg, cycles):
    j = JointHRL(cfg)
    n_envs = j.worker_env.num_envs
    cycle_steps = j.manager.n_steps * j.goal_every * n_envs
    rows = []

    def on_iter(it, model):
        fps = model.worker.num_timesteps / max(time.time() - model.worker.start_time, 1e-9)
        a, p = _mem()
        rows.append((it, fps, model.last_ep_rew_mean, a, p))

    j.learn(total_timesteps=cycles * cycle_steps, log_interval=10**9, on_iter=on_iter)

    nan = 0
    for v in tree_flatten(j.manager.policy.parameters()) + tree_flatten(j.worker.actor.parameters()):
        if not bool(mx.all(mx.isfinite(v[1])).item()):
            nan += 1
    rewards = [r[2] for r in rows]
    monotonic_decay = all(b < a for a, b in zip(rewards, rewards[1:]))
    fps_first, fps_last = rows[0][1], rows[-1][1]
    print(f"correctness: cycles={len(rows)} n_envs={n_envs} NaN_params={nan} "
          f"mean_win={rewards[0]:.4f}->{rewards[-1]:.4f} monotonic_decay={monotonic_decay}")
    print(f"  fps iter1={fps_first:.0f} iterN={fps_last:.0f} drift={(fps_last/fps_first-1)*100:+.1f}% "
          f"peak_mem={rows[-1][4]/1e6:.0f}MB")
    return {"nan": nan, "monotonic_decay": monotonic_decay, "fps_first": fps_first, "fps_last": fps_last}


def env_throughput(cfg, seconds=8.0):
    env = make_env(cfg, "continuous", cfg.get("hrl", {}).get("goal_dim", 3))
    n = env.num_envs
    obs, _ = env.reset()
    act = mx.zeros((n, 1))
    env.step(act)  # warm/compile
    n_steps = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        obs, r, d, info = env.step(act)
        n_steps += 1
    dt = time.time() - t0
    fps = n_steps * n / dt
    _, peak = _mem()
    print(f"env throughput: n_envs={n} {fps:.0f} env-steps/sec (collection only) peak_mem={peak/1e6:.0f}MB")
    return fps


def joint(cfg, cycles):
    j = JointHRL(cfg)
    n_envs = j.worker_env.num_envs
    cycle_steps = j.manager.n_steps * j.goal_every * n_envs
    t0 = time.time()
    j.learn(total_timesteps=cycles * cycle_steps, log_interval=10**9)
    dt = time.time() - t0
    steps = cycles * cycle_steps
    _, peak = _mem()
    print(f"joint: n_envs={n_envs} cycles={cycles} {steps/dt:.0f} steps/sec "
          f"({steps/1e6:.2f}M steps in {dt:.1f}s) peak_mem={peak/1e6:.0f}MB")
    return steps / dt


def sweep(cfg, envs):
    print(f"{'n_envs':>7} {'env_fps':>10} {'peak_mem_MB':>12}")
    base = dict(cfg.get("env", {}))
    for per_sym in envs:
        c = dict(cfg)
        e = dict(base)
        e["n_envs_per_symbol"] = per_sym
        c["env"] = e
        env = make_env(c, "continuous", 3)
        n = env.num_envs
        obs, _ = env.reset()
        act = mx.zeros((n, 1))
        env.step(act)
        n_steps, t0 = 0, time.time()
        while time.time() - t0 < 4.0:
            env.step(act)
            n_steps += 1
        fps = n_steps * n / (time.time() - t0)
        _, peak = _mem()
        print(f"{n:>7} {fps:>10.0f} {peak/1e6:>12.0f}")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["correctness", "env", "joint", "sweep"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    cfg = load(args.config) if args.config else load_smoke()
    if args.cmd == "correctness":
        correctness(cfg, args.cycles)
    elif args.cmd == "env":
        env_throughput(cfg, args.seconds)
    elif args.cmd == "joint":
        joint(cfg, args.cycles)
    elif args.cmd == "sweep":
        sweep(cfg, [1, 2, 4, 8, 16, 32])


if __name__ == "__main__":
    main()
