"""Vectorized per-symbol perpetuals trading environment (pure MLX, fused step).

Each environment instance trades one symbol with at most one open position
(flat / long / short) on isolated margin, with taker fees, entry slippage,
funding accrual, liquidation and bankruptcy truncation. The ``step`` is a
single ``mx.compile``-fused kernel with no Python loops; the state/time/RNG
threading matches ``dirty_mlx_ml.reinforcement`` so PPO and SAC consume it
directly (SB3-like ``reset`` / ``step`` plus the internal ``_step_fn``
contract).

Observation layout per env: features(F) + position one-hot(3) + account(3)
+ optional manager goal(goal_dim). State layout: balance, q, entry,
collateral, peak_equity + optional goal.

Reward is configurable via ``reward_mode``: ``"smoke"`` returns the per-step
log-equity change; ``"normal"`` subtracts a quadratic drawdown penalty (the
Ulcer term) so the policy maximizes a smooth "mountain-ridge" equity curve.
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx

from dirty_mlx_ml.reinforcement.spaces import Box, Discrete

log = logging.getLogger("trading")

EPS = 1e-8


class TradingEnv:
    def __init__(
        self,
        features,
        closes,
        *,
        action_space: str = "discrete",
        n_envs_per_symbol: int = 8,
        leverage: float = 5.0,
        fee_rate: float = 5e-4,
        slippage_bps: float = 1.0,
        funding_rate: float = 1e-4,
        maintenance_margin_rate: float = 5e-3,
        liquidation_fee_rate: float = 1e-2,
        min_collateral: float = 10.0,
        initial_balance: float = 1000.0,
        size_fraction: float = 1.0,
        side_threshold: float = 0.2,
        reward_mode: str = "smoke",
        drawdown_penalty: float = 1.0,
        goal_dim: int = 0,
        eval_every: int = 32,
        seed: int = 0,
    ):
        features = mx.array(features, dtype=mx.float32)
        closes = mx.array(closes, dtype=mx.float32)
        self.n_symbols = features.shape[0]
        self.T = features.shape[1]
        self.F = features.shape[2]
        self.n_envs_per_symbol = n_envs_per_symbol
        self.num_envs = self.n_symbols * n_envs_per_symbol

        self.closes_flat = mx.reshape(closes, (-1,))
        self.feats2d = mx.reshape(features, (self.n_symbols * self.T, self.F))
        sym_idx = mx.array(
            [s for s in range(self.n_symbols) for _ in range(n_envs_per_symbol)],
            dtype=mx.int32,
        )
        self.sym_off = sym_idx * self.T

        self.discrete = action_space == "discrete"
        self.goal_dim = int(goal_dim)
        self.leverage = float(leverage)
        self.fee_rate = float(fee_rate)
        self.slip = float(slippage_bps) / 1e4
        self.funding_rate = float(funding_rate)
        self.maintenance_margin_rate = float(maintenance_margin_rate)
        self.liquidation_fee_rate = float(liquidation_fee_rate)
        self.min_collateral = float(min_collateral)
        self.initial_balance = float(initial_balance)
        self.size_fraction = float(size_fraction)
        self.side_threshold = float(side_threshold)
        self.reward_mode = reward_mode
        self.drawdown_penalty = float(drawdown_penalty)
        self.eval_every = max(int(eval_every), 1)
        self._step_count = 0

        obs_dim = self.F + 6 + self.goal_dim
        self.observation_space = Box(
            low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype="float32"
        )
        self.single_observation_space = self.observation_space
        if self.discrete:
            self.action_space = Discrete(3)
        else:
            self.action_space = Box(
                low=mx.array([-1.0]), high=mx.array([1.0]), shape=(1,), dtype="float32"
            )
        self.single_action_space = self.action_space

        state_dim = 5 + self.goal_dim
        self._goal = mx.zeros((self.num_envs, self.goal_dim)) if self.goal_dim else None
        self._state = self._initial_state()
        self._steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((self.num_envs,), dtype=mx.bool_)
        self._key = mx.random.key(seed)
        self._step_fn = None
        log.info(
            "env: num_envs=%d n_symbols=%d T=%d obs_dim=%d action=%s goal_dim=%d reward=%s",
            self.num_envs, self.n_symbols, self.T, obs_dim,
            "discrete(3)" if self.discrete else "continuous(-1,1)", self.goal_dim, self.reward_mode,
        )

    def _initial_state(self):
        acct = mx.concatenate(
            [
                mx.full((self.num_envs, 1), self.initial_balance),
                mx.zeros((self.num_envs, 3)),
                mx.full((self.num_envs, 1), self.initial_balance),
            ],
            axis=1,
        )
        if self.goal_dim:
            return mx.concatenate([acct, self._goal], axis=1)
        return acct

    def set_goal(self, goal):
        g = mx.array(goal, dtype=mx.float32)
        if g.ndim == 1:
            g = mx.broadcast_to(g, (self.num_envs, self.goal_dim))
        self._goal = g
        self._state = mx.concatenate([self._state[:, :5], g], axis=1)

    def _obs(self, state, t_idx):
        t_idx = mx.minimum(t_idx, self.T - 1)
        idx = self.sym_off + t_idx
        feats = mx.take(self.feats2d, idx, axis=0)
        balance, q, entry, collateral = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        price = mx.take(self.closes_flat, idx)
        upnl = q * (price - entry)
        eq = balance + collateral + upnl
        pos = mx.stack([mx.abs(q) <= EPS, q > EPS, q < -EPS], axis=1).astype(mx.float32)
        acct = mx.stack(
            [
                eq / (self.initial_balance + EPS) - 1.0,
                upnl / (eq + EPS),
                collateral / (eq + EPS),
            ],
            axis=1,
        )
        obs = mx.concatenate([feats, pos, acct], axis=1)
        if self.goal_dim:
            obs = mx.concatenate([obs, state[:, 5:]], axis=1)
        return obs

    def reset(self, seed=None):
        if seed is not None:
            self._key = mx.random.key(seed)
        self._state = self._initial_state()
        self._steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._prev_done = mx.zeros((self.num_envs,), dtype=mx.bool_)
        return self._obs(self._state, self._steps), {}

    def _build_step(self):
        num_envs = self.num_envs
        T = self.T
        closes_flat = self.closes_flat
        feats2d = self.feats2d
        sym_off = self.sym_off
        goal_dim = self.goal_dim
        discrete = self.discrete
        leverage = self.leverage
        fee_rate = self.fee_rate
        slip = self.slip
        funding = self.funding_rate
        maint = self.maintenance_margin_rate
        liq_fee = self.liquidation_fee_rate
        min_col = self.min_collateral
        init_bal = self.initial_balance
        size_frac = self.size_fraction
        side_thr = self.side_threshold
        reward_mode = self.reward_mode
        dd_penalty = self.drawdown_penalty

        feats0 = mx.take(feats2d, sym_off, axis=0)
        pos0 = mx.concatenate([mx.ones((num_envs, 1)), mx.zeros((num_envs, 2))], axis=1)
        acct0 = mx.zeros((num_envs, 3))
        obs_static0 = mx.concatenate([feats0, pos0, acct0], axis=1)
        reset_acct = mx.concatenate(
            [
                mx.full((num_envs, 1), init_bal),
                mx.zeros((num_envs, 3)),
                mx.full((num_envs, 1), init_bal),
            ],
            axis=1,
        )

        def step(state, steps, prev_done, key, action):
            mask = prev_done
            t = mx.where(mask, mx.zeros_like(steps), steps)
            t_next = t + 1
            t_idx = mx.minimum(t, T - 1)
            tn_idx = mx.minimum(t_next, T - 1)

            price_prev = mx.take(closes_flat, sym_off + t_idx)
            price = mx.take(closes_flat, sym_off + tn_idx)

            balance = state[:, 0]
            q = state[:, 1]
            entry = state[:, 2]
            collateral = state[:, 3]
            peak_prev = state[:, 4]

            eq_prev = balance + collateral + q * (price_prev - entry)

            upnl = q * (price - entry)
            notional = mx.abs(q) * price

            liq = (mx.abs(q) > EPS) & ((collateral + upnl) <= maint * notional)
            balance = mx.where(
                liq,
                balance + mx.maximum(collateral + upnl - liq_fee * notional, 0.0),
                balance,
            )
            q = mx.where(liq, 0.0, q)
            entry = mx.where(liq, 0.0, entry)
            collateral = mx.where(liq, 0.0, collateral)

            if discrete:
                a = action.astype(mx.float32)
                side = mx.where(a == 1.0, 1.0, mx.where(a == 2.0, -1.0, 0.0))
                lev_frac = mx.full((num_envs,), size_frac)
            else:
                a = mx.clip(mx.reshape(action.astype(mx.float32), (num_envs,)), -1.0, 1.0)
                side = mx.where(mx.abs(a) > side_thr, mx.sign(a), 0.0)
                lev_frac = mx.maximum(mx.abs(a), 0.05)

            side = mx.where(liq, 0.0, side)

            side_cur = mx.sign(q)
            close = (mx.abs(q) > EPS) & (side_cur != side)
            balance = mx.where(
                close,
                balance + collateral + q * (price - entry) - fee_rate * mx.abs(q) * price,
                balance,
            )
            q = mx.where(close, 0.0, q)
            entry = mx.where(close, 0.0, entry)
            collateral = mx.where(close, 0.0, collateral)

            open_pos = (side != 0.0) & (mx.abs(q) <= EPS)
            avail = mx.maximum(balance, 0.0)
            lev_used = leverage * lev_frac
            notional_new = avail * lev_used
            fee_open = fee_rate * notional_new
            collateral_new = notional_new / lev_used - fee_open
            fill = price * (1.0 + side * slip)
            balance = mx.where(open_pos, balance - notional_new / lev_used, balance)
            collateral = mx.where(open_pos, collateral_new, collateral)
            q = mx.where(open_pos, side * notional_new / (fill + EPS), q)
            entry = mx.where(open_pos, fill, entry)

            balance = balance - funding * (q * price)

            upnl_end = q * (price - entry)
            eq_end = balance + collateral + upnl_end

            log_ret = mx.log(mx.maximum(eq_end, EPS)) - mx.log(mx.maximum(eq_prev, EPS))
            peak2 = mx.maximum(peak_prev, eq_end)
            if reward_mode == "normal":
                dd = (peak2 - eq_end) / (peak2 + EPS)
                reward = log_ret - dd_penalty * dd * dd
            else:
                reward = log_ret

            bankrupt = eq_end < min_col
            trunc_data = t_next >= (T - 1)
            truncated = trunc_data | bankrupt
            terminated = liq
            done = terminated | truncated

            state2 = mx.stack([balance, q, entry, collateral, peak2], axis=1)
            if goal_dim:
                goal = state[:, 5:]
                state2 = mx.concatenate([state2, goal], axis=1)

            feats = mx.take(feats2d, sym_off + tn_idx, axis=0)
            pos = mx.stack([mx.abs(q) <= EPS, q > EPS, q < -EPS], axis=1).astype(mx.float32)
            acct = mx.stack(
                [
                    eq_end / (init_bal + EPS) - 1.0,
                    upnl_end / (eq_end + EPS),
                    collateral / (eq_end + EPS),
                ],
                axis=1,
            )
            obs = mx.concatenate([feats, pos, acct], axis=1)
            if goal_dim:
                obs = mx.concatenate([obs, goal], axis=1)

            m = mask.astype(mx.float32)[:, None]
            reset_state = reset_acct if not goal_dim else mx.concatenate([reset_acct, goal], axis=1)
            reset_obs = obs_static0 if not goal_dim else mx.concatenate([obs_static0, goal], axis=1)
            state2 = state2 * (1.0 - m) + reset_state * m
            obs = obs * (1.0 - m) + reset_obs * m
            steps2 = mx.where(mask, mx.zeros_like(t_next), t_next)
            reward = mx.where(mask, mx.zeros_like(reward), reward)
            terminated = mx.where(mask, mx.zeros_like(terminated), terminated)
            truncated = mx.where(mask, mx.zeros_like(truncated), truncated)
            done = mx.where(mask, mx.zeros_like(done), done)

            return state2, steps2, done, key, obs, reward, done, truncated

        t0 = time.time()
        self._step_fn = mx.compile(step)
        log.debug("env: fused step compiled in %.2fs", time.time() - t0)

    def step(self, action):
        if self._step_fn is None:
            self._build_step()
        state, steps, prev_done, key, obs, reward, done, truncated = self._step_fn(
            self._state, self._steps, self._prev_done, self._key, action
        )
        self._step_count += 1
        if self._step_count % self.eval_every == 0:
            mx.eval(state, steps, prev_done, obs, reward, done, truncated)
        self._state = state
        self._steps = steps
        self._prev_done = prev_done
        self._key = key
        return obs, reward.astype(mx.float32), done, {"timeouts": truncated.astype(mx.float32)}

    def close(self):
        pass
