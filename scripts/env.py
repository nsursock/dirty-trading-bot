"""Vectorized per-symbol perpetuals trading environment (pure MLX, fused step).

Each environment instance trades one symbol with at most one open position
(flat / long / short) on isolated or cross margin, with platform-agnostic
taker fees (open 3 bps / close 6 bps), entry slippage, a daily holding-fee
buffer, and a dynamic collateral-retention liquidation threshold. The ``step``
is a single ``mx.compile``-fused kernel with no Python loops; the state/time/RNG
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
        highs=None,
        lows=None,
        action_space: str = "discrete",
        n_envs_per_symbol: int = 8,
        lev_min: float = 2.0,
        lev_max: float = 20.0,
        open_fee_rate: float = 3e-4,
        close_fee_rate: float = 6e-4,
        slippage_bps: float = 1.0,
        holding_fee_daily: float = 1.5e-4,
        bars_per_day: int = 288,
        liquidation_fee_rate: float = 3e-3,
        liq_threshold_base: float = 0.90,
        liq_threshold_floor: float = 0.67,
        liq_threshold_ref_lev: float = 2.0,
        liq_threshold_hi_lev: float = 150.0,
        liq_threshold_lo_lev: float = 1.0,
        liq_mark_impact: float = 0.005,
        min_collateral: float = 10.0,
        max_collateral: float = 10_000.0,
        risk_min: float = 0.01,
        risk_max: float = 0.05,
        take_profit: float = 0.0,
        stop_loss: float = 0.0,
        use_take_profit: bool = True,
        use_stop_loss: bool = True,
        initial_balance: float = 1000.0,
        size_fraction: float = 1.0,
        side_threshold: float = 0.2,
        trade_knob: float = 1.0,
        reward_mode: str = "smoke",
        drawdown_penalty: float = 1.0,
        reward_clip: float = 10.0,
        goal_dim: int = 0,
        eval_every: int = 32,
        enforce_goal: bool = False,
        margin_mode: str = "isolated",
        return_basis: str = "account",
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
        self.highs_flat = mx.reshape(highs, (-1,)) if highs is not None else None
        self.lows_flat = mx.reshape(lows, (-1,)) if lows is not None else None
        self.feats2d = mx.reshape(features, (self.n_symbols * self.T, self.F))
        sym_idx = mx.array(
            [s for s in range(self.n_symbols) for _ in range(n_envs_per_symbol)],
            dtype=mx.int32,
        )
        self.sym_off = sym_idx * self.T

        self.discrete = action_space == "discrete"
        self.goal_dim = int(goal_dim)
        self.lev_min = float(lev_min)
        self.lev_max = float(lev_max)
        self.open_fee_rate = float(open_fee_rate)
        self.close_fee_rate = float(close_fee_rate)
        self.slip = float(slippage_bps) / 1e4
        self.holding_fee_daily = float(holding_fee_daily)
        self.bars_per_day = int(bars_per_day)
        self.liquidation_fee_rate = float(liquidation_fee_rate)
        self.liq_threshold_base = float(liq_threshold_base)
        self.liq_threshold_floor = float(liq_threshold_floor)
        self.liq_threshold_ref_lev = float(liq_threshold_ref_lev)
        self.liq_threshold_hi_lev = float(liq_threshold_hi_lev)
        self.liq_threshold_lo_lev = float(liq_threshold_lo_lev)
        self.liq_mark_impact = float(liq_mark_impact)
        self.min_collateral = float(min_collateral)
        self.max_collateral = float(max_collateral)
        self.risk_min = float(risk_min)
        self.risk_max = float(risk_max)
        self.take_profit = float(take_profit)
        self.stop_loss = float(stop_loss)
        self.use_take_profit = bool(use_take_profit)
        self.use_stop_loss = bool(use_stop_loss)
        self.initial_balance = float(initial_balance)
        self.size_fraction = float(size_fraction)
        self.side_threshold = float(side_threshold)
        self.trade_knob = max(float(trade_knob), 1e-3)
        self.reward_mode = reward_mode
        self.drawdown_penalty = float(drawdown_penalty)
        self.reward_clip = float(reward_clip)
        self.eval_every = max(int(eval_every), 1)
        self.enforce_goal = bool(enforce_goal)
        if margin_mode not in ("isolated", "cross"):
            raise ValueError(f"margin_mode must be 'isolated' or 'cross', got {margin_mode!r}")
        self.margin_mode = margin_mode
        if return_basis not in ("account", "collateral"):
            raise ValueError(f"return_basis must be 'account' or 'collateral', got {return_basis!r}")
        self.return_basis = return_basis
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
        if self.margin_mode == "cross":
            eq = balance + upnl  # collateral is an allocation, not locked cash
        else:
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
        feats2d = self.feats2d
        sym_off = self.sym_off
        goal_dim = self.goal_dim
        discrete = self.discrete
        lev_min = self.lev_min
        lev_max = self.lev_max
        open_fee = self.open_fee_rate
        close_fee = self.close_fee_rate
        slip = self.slip
        funding = self.holding_fee_daily / max(int(self.bars_per_day), 1)
        liq_fee = self.liquidation_fee_rate
        thr_base = self.liq_threshold_base
        thr_floor = self.liq_threshold_floor
        thr_ref = self.liq_threshold_ref_lev
        thr_hi = self.liq_threshold_hi_lev
        thr_lo = self.liq_threshold_lo_lev
        # Fixed loss-fraction curve (independent of lev_max): 100% at thr_lo,
        # thr_base at thr_ref, thr_floor at thr_hi. Two linear segments.
        thr_slope_lo = (thr_base - 1.0) / max(thr_ref - thr_lo, 1e-6)
        thr_slope_hi = (thr_floor - thr_base) / max(thr_hi - thr_ref, 1e-6)
        min_col = self.min_collateral
        max_coll = self.max_collateral
        risk_min = self.risk_min
        risk_max = self.risk_max
        take_profit = self.take_profit
        stop_loss = self.stop_loss
        init_bal = self.initial_balance
        size_frac = self.size_fraction
        side_thr = self.side_threshold
        trade_knob = self.trade_knob
        reward_mode = self.reward_mode
        dd_penalty = self.drawdown_penalty
        reward_clip = self.reward_clip
        enforce_goal = self.enforce_goal
        return_basis = self.return_basis

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
        mx.eval(obs_static0, reset_acct)

        def step(state, steps, prev_done, key, action, price_prev, price, high, low, feats):
            mask = prev_done
            t = mx.where(mask, mx.zeros_like(steps), steps)
            t_next = t + 1

            balance = state[:, 0]
            q = state[:, 1]
            entry = state[:, 2]
            collateral = state[:, 3]
            peak_prev = state[:, 4]

            if self.margin_mode == "cross":
                # Whole-account equity backs the position; the collateral
                # column is only an allocation for sizing/leverage reporting.
                eq_prev = balance + q * (price_prev - entry)
            else:
                eq_prev = balance + collateral + q * (price_prev - entry)

            upnl = q * (price - entry)
            notional = mx.abs(q) * price

            tp_hit = mx.zeros((num_envs,), dtype=mx.bool_)
            sl_hit = mx.zeros((num_envs,), dtype=mx.bool_)
            exit_hit = mx.zeros((num_envs,), dtype=mx.bool_)
            sl_enabled = self.use_stop_loss and stop_loss > 0.0
            tp_enabled = self.use_take_profit and take_profit > 0.0
            if sl_enabled or tp_enabled:
                long = q > EPS
                short = q < -EPS
                sl_px = mx.where(long, entry * (1.0 - stop_loss), entry * (1.0 + stop_loss))
                tp_px = mx.where(long, entry * (1.0 + take_profit), entry * (1.0 - take_profit))
                sl_touch = mx.zeros((num_envs,), dtype=mx.bool_)
                tp_touch = mx.zeros((num_envs,), dtype=mx.bool_)
                if sl_enabled:
                    sl_touch = (long & (low <= entry * (1.0 - stop_loss))) | (
                        short & (high >= entry * (1.0 + stop_loss))
                    )
                if tp_enabled:
                    tp_touch = (long & (high >= entry * (1.0 + take_profit))) | (
                        short & (low <= entry * (1.0 - take_profit))
                    )
                sl_hit = sl_touch
                tp_hit = tp_touch & ~sl_touch
                exit_hit = sl_touch | tp_touch
                fill_exit = mx.where(sl_hit, sl_px, mx.where(tp_hit, tp_px, price))
                if self.margin_mode == "cross":
                    balance = mx.where(
                        exit_hit,
                        balance + q * (fill_exit - entry) - close_fee * mx.abs(q) * fill_exit,
                        balance,
                    )
                else:
                    balance = mx.where(
                        exit_hit,
                        balance + collateral + q * (fill_exit - entry) - close_fee * mx.abs(q) * fill_exit,
                        balance,
                    )
                q = mx.where(exit_hit, 0.0, q)
                entry = mx.where(exit_hit, 0.0, entry)
                collateral = mx.where(exit_hit, 0.0, collateral)

            # Dynamic liquidation threshold (isolated margin): liquidate once
            # the position has lost ``thr`` of its collateral (the PERP loss,
            # not the underlying move). Fixed curve, independent of lev_max:
            # 100% at 1x -> 90% at 2x -> 67% at 150x (then floored).
            lev_open = mx.abs(q) * entry / (collateral + EPS)
            thr = mx.where(
                lev_open <= thr_ref,
                1.0 + thr_slope_lo * (lev_open - thr_lo),
                thr_base + thr_slope_hi * (lev_open - thr_ref),
            )
            thr = mx.clip(thr, thr_floor, 1.0)
            if self.margin_mode == "cross":
                liq = (mx.abs(q) > EPS) & ((balance + q * (price - entry)) <= (1.0 - thr) * collateral)
                balance = mx.where(
                    liq,
                    mx.maximum(balance - collateral - liq_fee * notional, 0.0),
                    balance,
                )
            else:
                liq = (mx.abs(q) > EPS) & ((collateral + q * (price - entry)) <= (1.0 - thr) * collateral)
                # Full collateral loss on liquidation: the locked margin is
                # forfeited (never returned to cash). The liquidation fee is
                # charged on top of the forfeit, floored at zero so isolated
                # bankruptcy truncation keeps the account non-negative.
                balance = mx.where(
                    liq,
                    mx.maximum(balance - liq_fee * notional, 0.0),
                    balance,
                )
            q = mx.where(liq, 0.0, q)
            entry = mx.where(liq, 0.0, entry)
            collateral = mx.where(liq, 0.0, collateral)

            if discrete:
                a = action.astype(mx.float32)
                side = mx.where(a == 1.0, 1.0, mx.where(a == 2.0, -1.0, 0.0))
                t = mx.full((num_envs,), size_frac)
            else:
                a = mx.clip(mx.reshape(action.astype(mx.float32), (num_envs,)), -1.0, 1.0)
                eff_thr = side_thr / trade_knob
                side = mx.where(mx.abs(a) > eff_thr, mx.sign(a), 0.0)
                t = mx.abs(a)

            side = mx.where(liq | exit_hit, 0.0, side)

            if goal_dim >= 3 and enforce_goal:
                goal = state[:, 5:]
                g_flat, g_long, g_short = goal[:, 0], goal[:, 1], goal[:, 2]
                side = mx.where(g_long, mx.maximum(side, 0.0), side)
                side = mx.where(g_short, mx.minimum(side, 0.0), side)
                side = mx.where(g_flat, 0.0, side)

            side_cur = mx.sign(q)
            close = (mx.abs(q) > EPS) & (side_cur != side)
            if self.margin_mode == "cross":
                balance = mx.where(
                    close,
                    balance + q * (price - entry) - close_fee * mx.abs(q) * price,
                    balance,
                )
            else:
                balance = mx.where(
                    close,
                    balance + collateral + q * (price - entry) - close_fee * mx.abs(q) * price,
                    balance,
                )
            q = mx.where(close, 0.0, q)
            entry = mx.where(close, 0.0, entry)
            collateral = mx.where(close, 0.0, collateral)

            open_pos = (side != 0.0) & (mx.abs(q) <= EPS)
            risk_frac = risk_min + t * (risk_max - risk_min)
            lev_used = lev_min + t * (lev_max - lev_min)
            if self.margin_mode == "cross":
                # Size off total account equity so unrealized gains compound.
                size_base = balance + q * (price - entry)
            else:
                size_base = balance
            collateral_new = mx.clip(risk_frac * size_base, min_col, max_coll)
            notional_new = collateral_new * lev_used
            fee_open = open_fee * notional_new
            fill = price * (1.0 + side * slip)
            if self.margin_mode == "cross":
                # Margin is not locked: the account pays only the entry fee,
                # the whole equity stays available as backing.
                balance = mx.where(open_pos, balance - fee_open, balance)
            else:
                balance = mx.where(open_pos, balance - collateral_new - fee_open, balance)
            collateral = mx.where(open_pos, collateral_new, collateral)
            q = mx.where(open_pos, side * notional_new / (fill + EPS), q)
            entry = mx.where(open_pos, fill, entry)

            balance = balance - funding * (q * price)

            upnl_end = q * (price - entry)
            if self.margin_mode == "cross":
                eq_end = balance + upnl_end
            else:
                eq_end = balance + collateral + upnl_end

            peak2 = mx.maximum(peak_prev, eq_end)
            if return_basis == "collateral":
                # Per-step return on the deployed collateral (ROC); flat
                # accounts (no collateral at step start) earn zero return.
                pnl = eq_end - eq_prev
                roc = mx.where(
                    collateral > EPS,
                    mx.log(mx.maximum(1.0 + pnl / mx.maximum(collateral, EPS), EPS)),
                    mx.zeros_like(eq_end),
                )
                log_ret = roc
            else:
                log_ret = mx.log(mx.maximum(eq_end, EPS)) - mx.log(mx.maximum(eq_prev, EPS))
            if reward_mode == "normal":
                dd = (peak2 - eq_end) / (peak2 + EPS)
                reward = log_ret - dd_penalty * dd * dd
            else:
                reward = log_ret
            reward = mx.clip(reward, -reward_clip, reward_clip)

            bankrupt = eq_end < min_col
            trunc_data = t_next >= (T - 1)
            truncated = trunc_data | bankrupt
            terminated = liq
            done = terminated | truncated

            state2 = mx.stack([balance, q, entry, collateral, peak2], axis=1)
            if goal_dim:
                goal = state[:, 5:]
                state2 = mx.concatenate([state2, goal], axis=1)

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

            exit_flag = mx.where(sl_hit, 2, mx.where(tp_hit, 1, mx.where(liq, 3, 0))).astype(mx.int32)
            exit_flag = mx.where(mask, 0, exit_flag)

            return state2, steps2, done, key, obs, reward, done, truncated, exit_flag

        t0 = time.time()
        self._step_fn = mx.compile(step)
        log.debug("env: fused step compiled in %.2fs", time.time() - t0)

    def step(self, action):
        if self._step_fn is None:
            self._build_step()
        t = mx.where(self._prev_done, mx.zeros_like(self._steps), self._steps)
        t_next = t + 1
        tn_idx = mx.minimum(t_next, self.T - 1)
        price_prev = mx.take(self.closes_flat, self.sym_off + mx.minimum(t, self.T - 1))
        price = mx.take(self.closes_flat, self.sym_off + tn_idx)
        high = price
        low = price
        if self.highs_flat is not None:
            high = mx.take(self.highs_flat, self.sym_off + tn_idx)
            low = mx.take(self.lows_flat, self.sym_off + tn_idx)
        feats = mx.take(self.feats2d, self.sym_off + tn_idx, axis=0)
        state, steps, prev_done, key, obs, reward, done, truncated, exit_flag = self._step_fn(
            self._state, self._steps, self._prev_done, self._key, action,
            price_prev, price, high, low, feats,
        )
        self._step_count += 1
        if self._step_count % self.eval_every == 0:
            mx.eval(state, steps, prev_done, obs, reward, done, truncated, exit_flag)
        self._state = state
        self._steps = steps
        self._prev_done = prev_done
        self._key = key
        return obs, reward.astype(mx.float32), done, {
            "timeouts": truncated.astype(mx.float32),
            "exit": exit_flag,
        }

    def close(self):
        pass
