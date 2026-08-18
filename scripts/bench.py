"""Benchmark / instrumentation harness for the joint-HRL milestones (consensus.md).

Subcommands:
  correctness   joint loop N cycles @ cfg envs: per-cycle FPS / reward / memory + NaN scan
  env           env-only collection throughput (no gradient calls)
  joint         joint loop (rollout + PPO + SAC) steps/sec + peak memory over N cycles
  sweep         env-throughput vs num_envs, doubling until FPS plateaus,
                MLX nears RAM, or the system starts swapping
    sweep_joint    env AND joint-train throughput vs num_envs (same sweep)
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import subprocess
import time

import mlx.core as mx
from mlx.utils import tree_flatten

from agents import JointHRL, make_env
from config import load, load_smoke
from data import SYMBOLS

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


def _sysctl_swap() -> float:
    """Used swap bytes on macOS (0.0 when vm.swapusage is unavailable)."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode()
        m = re.search(r"used\s*=\s*([\d.]+)([KMG])", out)
        if not m:
            return 0.0
        mult = {"K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]
        return float(m.group(1)) * mult
    except (OSError, subprocess.SubprocessError):
        return 0.0


def _env_fps(cfg, seconds):
    """Collection-only env throughput: returns (num_envs, fps, peak_mb, active_mb)."""
    mx.reset_peak_memory()
    mx.clear_cache()
    env = make_env(cfg, "continuous", cfg.get("hrl", {}).get("goal_dim", 3))
    n = env.num_envs
    obs, _ = env.reset()
    act = mx.zeros((n, 1))
    env.step(act)
    mx.clear_cache()
    n_steps, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        obs, r, d, info = env.step(act)
        mx.eval(obs)
        n_steps += 1
    fps = n_steps * n / (time.time() - t0)
    a, p = _mem()
    return n, fps, p / 1e6, a / 1e6


def _joint_fps(cfg, cycles):
    """Joint rollout + PPO + SAC throughput: returns (fps, peak_mb)."""
    mx.reset_peak_memory()
    mx.clear_cache()
    j = JointHRL(cfg)
    n = j.worker_env.num_envs
    cycle_steps = j.manager.n_steps * j.goal_every * n
    total = max(int(cycles * cycle_steps), 1)
    t0 = time.time()
    j.learn(total_timesteps=total, log_interval=10**9, log_every=0)
    dt = time.time() - t0
    _, p = _mem()
    del j
    gc.collect()
    mx.clear_cache()
    return total / dt, p / 1e6


def sweep(cfg, max_per_sym=2048, seconds=4.0, joint=False, cycles=1):
    """Throughput while doubling num_envs until a stop condition hits.

    Doubles ``n_envs_per_symbol`` each iteration and reports env-only FPS, and
    (when ``joint``) the joint rollout + PPO + SAC training FPS at the same env
    count. Stops when the FPS gain for a 2x env increase falls below
    ``PLATEAU_GAIN`` (GPU/CPU saturation), when the MLX footprint exceeds
    ``MEM_FRAC`` of physical RAM, or when used swap grows > ``SWAP_GROW`` since
    start. The env observation is materialized every step (``mx.eval``) so lazy
    MLX arrays do not hide the real per-env footprint; the allocator cache and
    peak counter are reset before each measurement so each column reflects that
    config alone.
    """
    PLATEAU_GAIN = 1.4
    MEM_FRAC = 0.7
    SWAP_GROW = 1e9
    total_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    base = dict(cfg.get("env", {}))
    cols = ("n_envs", "env_fps", "env_peak_MB", "train_fps", "train_peak_MB", "swap_used_MB") \
        if joint else ("n_envs", "env_fps", "peak_mem_MB", "active_MB", "swap_used_MB")
    print(" ".join(f"{c:>12}" for c in cols))
    swap0 = _sysctl_swap()
    prev_env_fps = None
    per_sym = 1
    while per_sym <= max_per_sym:
        c = dict(cfg)
        e = dict(base)
        e["n_envs_per_symbol"] = per_sym
        c["env"] = e
        n = 0
        try:
            n, env_fps, env_peak, env_active = _env_fps(c, seconds)
        except RuntimeError as ex:
            print(f"{'n_envs':>12}  err: {str(ex)[:48]}")
            break
        train_fps, train_peak = None, None
        if joint:
            try:
                train_fps, train_peak = _joint_fps(c, cycles)
            except RuntimeError as ex:
                train_fps, train_peak = None, None
                print(f"{n:>12} {'env_only':>12} joint err: {str(ex)[:40]}")
                break
        swap = _sysctl_swap()
        mx_footprint = (train_peak if joint else env_active) * 1e6
        if joint:
            print(f"{n:>12} {env_fps:>12.0f} {env_peak:>12.0f} {train_fps:>12.0f} "
                  f"{train_peak:>12.0f} {swap/1e6:>12.0f}")
        else:
            print(f"{n:>12} {env_fps:>12.0f} {env_peak:>12.0f} {env_active:>12.0f} {swap/1e6:>12.0f}")
        growth = env_fps / prev_env_fps if prev_env_fps else None
        if growth is not None and growth < PLATEAU_GAIN:
            print(f"{'':>12}  plateau: env fps rose {growth:.2f}x for a 2x env increase -> stop")
            break
        if mx_footprint > MEM_FRAC * total_ram:
            print(f"{'':>12}  stop: MLX footprint {mx_footprint/1e9:.1f}GB = {mx_footprint/total_ram*100:.0f}% of RAM")
            break
        if swap - swap0 > SWAP_GROW:
            print(f"{'':>12}  stop: swap grew {(swap-swap0)/1e6:.0f}MB -> swapping")
            break
        prev_env_fps = env_fps
        per_sym *= 2


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["correctness", "env", "joint", "sweep", "sweep_joint"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--max-per-sym", type=int, default=2048,
                    help="sweep: stop doubling once per-symbol envs exceed this")
    args = ap.parse_args()
    cfg = load(args.config) if args.config else load_smoke()
    if args.cmd == "correctness":
        correctness(cfg, args.cycles)
    elif args.cmd == "env":
        env_throughput(cfg, args.seconds)
    elif args.cmd == "joint":
        joint(cfg, args.cycles)
    elif args.cmd == "sweep":
        sweep(cfg, max_per_sym=args.max_per_sym, seconds=args.seconds)
    elif args.cmd == "sweep_joint":
        sweep(cfg, max_per_sym=args.max_per_sym, seconds=args.seconds,
              joint=True, cycles=args.cycles)


if __name__ == "__main__":
    main()
