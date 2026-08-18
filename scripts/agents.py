"""PPO Manager + SAC Worker (HRL) on dirty-mlx-ml, SB3 API parity.

``JointHRL`` trains the two tiers *jointly* on a shared timeline: the discrete
PPO manager picks a directional goal every ``goal_every`` worker steps, the
continuous SAC worker executes with that goal appended to its observation, and
both policies update inside the same loop. Building blocks (policy nets,
buffers, optimizers, compiled update steps) come from ``dirty_mlx_ml``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import yaml
from dirty_mlx_ml.reinforcement import PPO, SAC, VecNormalize
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

from data import SYMBOLS, build_high_view, generate, mgr_obs
from env import TradingEnv

log = logging.getLogger("trading")


def _plain(obj):
    """Recursively unwrap Config -> plain dict/list so yaml can dump it."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def config_hash(cfg) -> str:
    """Stable SHA-256 fingerprint of a resolved config dict."""
    return hashlib.sha256(yaml.safe_dump(_plain(cfg), sort_keys=True).encode()).hexdigest()


def _git_info() -> dict:
    try:
        root = Path(__file__).resolve().parents[1]
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        return {"HEAD": head.stdout.strip() if head.returncode == 0 else "", "dirty": bool(dirty.stdout.strip())}
    except Exception:
        return {"HEAD": "", "dirty": False}


def _slice_symbols(n: int) -> dict:
    return dict(list(SYMBOLS.items())[: n])


def build_bundle(cfg):
    d = cfg.get("data", {})
    symbols = _slice_symbols(d.get("n_symbols", 4))
    tf = d.get("timeframes", {})
    return generate(
        symbols=symbols,
        n_steps=d.get("n_steps", 400),
        seed=cfg.get("seed", 42),
        low_tf=tf.get("low", 5),
        high_tf=tf.get("high", 240),
        regime=d.get("regime", "bull"),
    )


def _env_kwargs(cfg) -> dict:
    e = dict(cfg.get("env", {}))
    r = dict(cfg.get("reward", {}))
    rs = dict(cfg.get("returns", {}))
    e["reward_mode"] = r.get("mode", "smoke")
    e["drawdown_penalty"] = r.get("drawdown_penalty", 1.0)
    e["return_basis"] = rs.get("basis", "account")
    return e


def make_env(cfg, action_space: str, goal_dim: int = 0, bundle=None, trade_knob=None) -> TradingEnv:
    if bundle is None:
        bundle = build_bundle(cfg)
    kw = _env_kwargs(cfg)
    kw["action_space"] = action_space
    kw["goal_dim"] = goal_dim
    if trade_knob is not None:
        kw["trade_knob"] = trade_knob
    return TradingEnv(
        bundle.features, bundle.ohlcv.closes,
        highs=bundle.ohlcv.highs, lows=bundle.ohlcv.lows,
        seed=cfg.get("seed", 42), **kw,
    )


def make_ppo(env, cfg, log_dir=None) -> PPO:
    c = dict(cfg.get("manager", {}))
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=c.get("learning_rate", 3e-4),
        n_steps=c.get("n_steps", 128),
        batch_size=c.get("batch_size", 32),
        n_epochs=c.get("n_epochs", 4),
        gamma=c.get("gamma", 0.99),
        gae_lambda=c.get("gae_lambda", 0.95),
        clip_range=c.get("clip_range", 0.2),
        ent_coef=c.get("ent_coef", 0.0),
        vf_coef=c.get("vf_coef", 0.5),
        seed=cfg.get("seed"),
        log_dir=log_dir,
        policy_kwargs={"net_arch": c.get("net_arch", [64, 64])},
    )


def make_sac(env, cfg, log_dir=None) -> SAC:
    c = dict(cfg.get("worker", {}))
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=c.get("learning_rate", 3e-4),
        buffer_size=c.get("buffer_size", 10_000),
        learning_starts=c.get("learning_starts", 256),
        batch_size=c.get("batch_size", 128),
        tau=c.get("tau", 0.005),
        gamma=c.get("gamma", 0.99),
        train_freq=c.get("train_freq", 1),
        gradient_steps=c.get("gradient_steps", 1),
        ent_coef=c.get("ent_coef", "auto"),
        seed=cfg.get("seed"),
        log_dir=log_dir,
        policy_kwargs={"net_arch": c.get("net_arch", [128, 128])},
    )


def _one_hot(a, n: int) -> mx.array:
    return mx.equal(mx.arange(n), a[:, None]).astype(mx.float32)


class JointHRL:
    """Joint two-tier trainer: PPO manager (goals) + SAC worker (execution)."""

    def _mgr_obs(self, worker_obs, low_steps):
        """Manager sees the last *completed* high-TF window + account state.

        The manager view is built on ``build_high_view`` so it never reads
        the high-TF bar that is still forming (causal, no lookahead).
        """
        F = self.worker_env.F
        feats = mgr_obs(self.mgr_high_feats2d, worker_obs[:, F : F + 6],
                        low_steps, self.sym_off_high, self.T_hi, self.n_resample)
        return feats

    def __init__(self, cfg, log_dir=None, config_path=None):
        self.cfg = cfg
        self.log_dir = log_dir
        self.bundle = build_bundle(cfg)
        h = dict(cfg.get("hrl", {}))
        self.goal_dim = h.get("goal_dim", 3)
        self.goal_every = h.get("goal_every", 4)
        n = dict(cfg.get("norm", {}))
        self.worker_env = VecNormalize(
            make_env(cfg, "continuous", self.goal_dim, self.bundle, trade_knob=1.0),
            norm_obs=n.get("norm_obs", True),
            norm_reward=n.get("norm_reward", True),
            clip_obs=n.get("clip_obs", 10.0),
            clip_reward=n.get("clip_reward", 10.0),
            gamma=n.get("gamma", 0.99),
        )
        self.mgr_env = make_env(cfg, "discrete", 0, self.bundle)
        self.obs_mgr_dim = self.mgr_env.observation_space.shape[0]
        self.config_path = str(config_path) if config_path is not None else None
        S, T_hi, F_hi = self.bundle.high_features.shape
        W = self.worker_env.n_envs_per_symbol
        sym_idx = mx.array([s for s in range(S) for _ in range(W)], dtype=mx.int32)
        self.mgr_high_feats2d = build_high_view(self.bundle.high_features)
        self.sym_off_high = sym_idx * T_hi
        self.T_hi = T_hi
        self.n_resample = self.bundle.n_resample
        self.manager = make_ppo(self.mgr_env, cfg, log_dir)
        self.worker = make_sac(self.worker_env, cfg, log_dir)
        mgr_cfg = dict(cfg.get("manager", {}))
        wrk_cfg = dict(cfg.get("worker", {}))
        log.info(
            "manager: PPO discrete obs_dim=%d n_steps=%d net_arch=%s lr=%s",
            self.obs_mgr_dim, mgr_cfg.get("n_steps"), mgr_cfg.get("net_arch"), mgr_cfg.get("learning_rate"),
        )
        log.info(
            "worker: SAC continuous obs_dim=%d buffer=%s batch=%s net_arch=%s lr=%s",
            self.worker_env.observation_space.shape[0], wrk_cfg.get("buffer_size"),
            wrk_cfg.get("batch_size"), wrk_cfg.get("net_arch"), wrk_cfg.get("learning_rate"),
        )
        obs_dim = self.worker_env.observation_space.shape[0]
        buf_bytes = wrk_cfg.get("buffer_size", 10_000) * (2 * obs_dim + 1 + 3) * 4
        log.info("worker: SAC replay buffer = %d transitions ~= %.1f MB",
                 wrk_cfg.get("buffer_size", 10_000), buf_bytes / 1e6)

    def learn(self, total_timesteps=None, log_interval=1, on_iter=None, checkpoint_every=300, log_every=0):
        cfg = self.cfg
        total = total_timesteps if total_timesteps is not None else cfg.get("train", {}).get(
            "total_timesteps", 4096
        )
        ppo, sac = self.manager, self.worker
        env = self.worker_env
        n_envs = env.num_envs
        n_steps = ppo.n_steps
        goal_every = self.goal_every
        goal_dim = self.goal_dim
        obs_mgr_dim = self.obs_mgr_dim

        if env._step_fn is None:
            env._build_step()

        ppo.start_time = sac.start_time = time.time()
        ppo._num_timesteps_at_start = sac._num_timesteps_at_start = 0
        ppo.num_timesteps = sac.num_timesteps = 0
        ppo._ep_rew = mx.zeros((n_envs,))
        ppo._ep_len = mx.zeros((n_envs,))
        ppo._roll_rew_sum = ppo._roll_len_sum = ppo._roll_ep_count = mx.array(0.0)

        worker_obs = env.reset()[0]
        mgr_obs = self._mgr_obs(worker_obs, env._steps)
        mgr_starts = mx.ones((n_envs,))
        log.debug("worker env reset: obs=%s", tuple(worker_obs.shape))

        cycle_steps = n_steps * goal_every * n_envs
        log.info(
            "joint training: total=%d num_envs=%d n_steps=%d goal_every=%d cycle_steps=%d",
            total, n_envs, n_steps, goal_every, cycle_steps,
        )
        pbar = tqdm(total=total, desc="train", unit="env-step")
        iteration = 0
        sac_started = False
        last_ckpt = time.time()
        if log_every <= 0:
            log_every = cycle_steps  # fall back to once-per-cycle
        next_log = log_every
        while sac.num_timesteps < total:
            ppo.buffer.reset()
            cycle_win = mx.zeros((n_envs,))
            obs_l, act_l, rew_l, st_l, val_l, lp_l = [], [], [], [], [], []
            for i in range(n_steps):
                ppo._policy_key, k_act = mx.random.split(ppo._policy_key)
                goal, value, log_prob = ppo.policy.get_action(mgr_obs, key=k_act)
                env.set_goal(_one_hot(goal, goal_dim))

                win = mx.zeros((n_envs,))
                window_done = mx.zeros((n_envs,), dtype=mx.bool_)
                for _ in range(goal_every):
                    obs_prev = worker_obs
                    w_act = sac._sample_action(obs_prev, random=sac.num_timesteps < sac.learning_starts)
                    worker_obs, r, d, info = env.step(w_act)
                    sac.num_timesteps += n_envs
                    truncated = info.get("timeouts", mx.zeros((n_envs,)))
                    sac._update_ep_stats(r, d)
                    sac.replay.add(obs_prev, worker_obs, w_act, r, d.astype(mx.float32), truncated)
                    win = win + r
                    window_done = window_done | d

                if sac.num_timesteps >= sac.learning_starts:
                    if not sac_started:
                        log.info("worker: SAC learning_starts reached (timestep=%d)", sac.num_timesteps)
                        sac_started = True
                    sac.train(sac.gradient_steps, sac.batch_size)

                pbar.update(goal_every * n_envs)
                cycle_win = cycle_win + win
                if (i + 1) % max(n_steps // 10, 1) == 0:
                    log.debug(
                        "window=%d/%d timesteps=%d win_mean=%.6f",
                        i + 1, n_steps, sac.num_timesteps, float(mx.mean(win)),
                    )

                obs_l.append(mgr_obs)
                act_l.append(mx.reshape(goal.astype(mx.float32), (n_envs, 1)))
                rew_l.append(win)
                st_l.append(mgr_starts)
                val_l.append(value)
                lp_l.append(log_prob)

                finished = window_done.astype(mx.float32)
                ep_rew = ppo._ep_rew + win
                ppo._roll_rew_sum = ppo._roll_rew_sum + mx.sum(mx.where(finished > 0.5, ep_rew, 0.0))
                ppo._roll_len_sum = ppo._roll_len_sum + mx.sum(
                    mx.where(finished > 0.5, ppo._ep_len + 1.0, 0.0)
                )
                ppo._roll_ep_count = ppo._roll_ep_count + mx.sum(finished)
                ppo._ep_rew = ep_rew * (1.0 - finished)
                ppo._ep_len = (ppo._ep_len + 1.0) * (1.0 - finished)

                mgr_starts = finished
                mgr_obs = self._mgr_obs(worker_obs, env._steps)

                if sac.num_timesteps >= next_log:
                    ppo.dump_logs(iteration)
                    sac.dump_logs(iteration)
                    next_log += log_every

            ppo.buffer.obs = mx.stack(obs_l)
            ppo.buffer.actions = mx.stack(act_l)
            ppo.buffer.rewards = mx.stack(rew_l)
            ppo.buffer.episode_starts = mx.stack(st_l)
            ppo.buffer.values = mx.stack(val_l)
            ppo.buffer.log_probs = mx.stack(lp_l)
            ppo.buffer.pos = n_steps
            ppo.buffer.full = True
            ppo.num_timesteps += n_steps * n_envs
            last_values = ppo.policy.forward(mgr_obs)[1]
            ppo.buffer.compute_returns_and_advantage(last_values, mgr_starts)
            ppo.train()

            iteration += 1
            self.last_ep_rew_mean = float(mx.sum(cycle_win)) / max(n_steps * n_envs, 1)
            if on_iter is not None:
                on_iter(iteration, self)
            if log_interval and iteration % log_interval == 0:
                elapsed = time.time() - ppo.start_time
                fps = sac.num_timesteps / max(elapsed, 1e-9)
                ppo_m = {k.split("/")[-1]: v for k, v in ppo.logger._vals.items() if k.startswith("train/")}
                sac_m = {}
                if sac._train_sums and sac._train_count:
                    n = max(sac._train_count, 1)
                    sac_m = {
                        k: float(mx.array(sac._train_sums[k]).item()) / n
                        for k in ("actor_loss", "critic_loss", "ent_coef")
                        if k in sac._train_sums
                    }
                log.info(
                    "cycle=%d timesteps=%d fps=%.0f mean_win=%.6f "
                    "mem_active=%.0fMB mem_peak=%.0fMB ppo=%s sac=%s",
                    iteration, sac.num_timesteps, fps, self.last_ep_rew_mean,
                    mx.get_active_memory() / 1e6, mx.get_peak_memory() / 1e6, ppo_m, sac_m,
                )

            if self.log_dir and time.time() - last_ckpt >= checkpoint_every:
                self.save(self.log_dir)
                log.info("checkpoint saved: timesteps=%d", sac.num_timesteps)
                last_ckpt = time.time()

        pbar.close()
        log.info("training done: total_timesteps=%d", sac.num_timesteps)
        ppo.dump_logs(iteration)
        sac.dump_logs(iteration)
        ppo.logger.close()
        sac.logger.close()
        return self

    def save(self, dirpath, config_path=None):
        os.makedirs(dirpath, exist_ok=True)
        mx.save_safetensors(
            os.path.join(dirpath, "manager_policy.safetensors"),
            dict(tree_flatten(self.manager.policy.parameters())),
        )
        mx.save_safetensors(
            os.path.join(dirpath, "worker_actor.safetensors"),
            dict(tree_flatten(self.worker.actor.parameters())),
        )
        ns = self.worker_env.norm_state
        flat = {}
        for k in ("obs_mean", "obs_var", "obs_count", "ret_mean", "ret_var", "ret_count"):
            v = mx.array(ns[k])
            flat[k] = v.reshape(-1) if v.ndim == 0 else v
        mx.save_safetensors(os.path.join(dirpath, "worker_norm.safetensors"), flat)

        m_cfg = dict(self.cfg.get("manager", {}))
        w_cfg = dict(self.cfg.get("worker", {}))
        cp = config_path or self.config_path
        manifest = {
            "config_path": str(cp) if cp is not None else None,
            "config_hash": config_hash(self.cfg),
            "seed": self.cfg.get("seed"),
            "goal_dim": self.goal_dim,
            "goal_every": self.goal_every,
            "n_resample": self.n_resample,
            "manager_obs_dim": self.obs_mgr_dim,
            "worker_obs_dim": self.worker_env.observation_space.shape[0],
            "manager": {
                "n_steps": m_cfg.get("n_steps"),
                "net_arch": m_cfg.get("net_arch"),
                "learning_rate": m_cfg.get("learning_rate"),
            },
            "worker": {
                "net_arch": w_cfg.get("net_arch"),
                "learning_rate": w_cfg.get("learning_rate"),
                "buffer_size": w_cfg.get("buffer_size"),
            },
            "git": _git_info(),
            "saved_at": datetime.now().isoformat(),
        }
        with open(os.path.join(dirpath, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        return dirpath

    def load(self, dirpath):
        m = mx.load(os.path.join(dirpath, "manager_policy.safetensors"))
        self.manager.policy.update(tree_unflatten(list(m.items())))
        w = mx.load(os.path.join(dirpath, "worker_actor.safetensors"))
        self.worker.actor.update(tree_unflatten(list(w.items())))
        p = os.path.join(dirpath, "worker_norm.safetensors")
        if os.path.exists(p):
            d = mx.load(p)
            scalars = {"obs_count", "ret_mean", "ret_var", "ret_count"}
            for k, v in d.items():
                self.worker_env.norm_state[k] = v[0] if k in scalars else v
        return self


def train_hrl(cfg, log_dir=None):
    return JointHRL(cfg, log_dir).learn()
