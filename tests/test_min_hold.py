"""min_hold_bars commitment-floor tests.

The env honors flat/flip jitter closes only after a position has been held
for ``min_hold_bars`` full bars (opening bar counts as 1). Stop-loss,
take-profit and liquidation exit via their own intrabar path and must fire
immediately regardless of the floor.
"""

import mlx.core as mx
import numpy as np
import pytest
from env import TradingEnv


def _flat_series(n=10):
    closes = np.full((n,), 100.0, dtype=np.float32)
    highs = np.full((n,), 100.0, dtype=np.float32)
    lows = np.full((n,), 100.0, dtype=np.float32)
    return closes, highs, lows


def _make_env(min_hold_bars, **over):
    closes, highs, lows = _flat_series()
    kw = dict(
        n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
        lev_min=2.0, lev_max=20.0, risk_min=0.05, risk_max=0.05,
        take_profit=0.0, stop_loss=0.0,
        use_take_profit=False, use_stop_loss=False,
        trade_knob=1.0, open_fee_rate=0.0, close_fee_rate=0.0,
        holding_fee_daily=0.0, margin_mode="isolated", seed=1,
        min_hold_bars=min_hold_bars,
    )
    kw.update(over)
    feats = mx.zeros((1, closes.size, 4))
    return TradingEnv(feats,
                      mx.array(np.expand_dims(closes, axis=0)),
                      highs=mx.array(np.expand_dims(highs, axis=0)),
                      lows=mx.array(np.expand_dims(lows, axis=0)), **kw)


def _q(env):
    return float(np.asarray(env._state)[0, 1])


def _crash_series():
    closes = np.array([100.0, 99.0, 97.0, 94.0, 90.0, 86.0,
                       81.0, 76.0, 70.0, 64.0, 58.0, 52.0], dtype=np.float32)
    highs = np.maximum(closes, np.roll(closes, 1) * 1.01).astype(np.float32)
    lows = np.minimum(closes, np.roll(closes, 1) * 0.99).astype(np.float32)
    return closes, highs, lows


def test_jitter_close_blocked_until_min_hold():
    # Open long, then ask flat/flip every bar from bar 2 on. The position must
    # survive bars 2-3 (age 1, 2 < 3) and only be closed at bar 4 (age 3).
    env = _make_env(min_hold_bars=3)
    env.reset()
    env.step(mx.array([[1.0]]))          # open long
    assert _q(env) > 0.0
    for _ in range(2):                   # bars 2,3: jitter close denied
        env.step(mx.array([[-1.0]]))
        assert _q(env) > 0.0, "position closed before min_hold elapsed"
    env.step(mx.array([[-1.0]]))         # bar 4: floor reached -> close + flip
    assert _q(env) < 0.0


def test_default_min_hold_zero_closes_immediately():
    # Backward compatibility: without a floor, the jitter close is honored
    # on the very next bar.
    env = _make_env(min_hold_bars=0)
    env.reset()
    env.step(mx.array([[1.0]]))
    env.step(mx.array([[-1.0]]))
    assert _q(env) < 0.0


def test_min_hold_two_blocks_next_bar_flip():
    # Opening bar counts as age 1, so with min_hold_bars=2 a flip on the very
    # next step (age 1) is denied, but one bar later (age 2) it is honored.
    env = _make_env(min_hold_bars=2)
    env.reset()
    env.step(mx.array([[1.0]]))
    env.step(mx.array([[-1.0]]))
    assert _q(env) > 0.0, "flip not blocked for min_hold_bars=2"
    env.step(mx.array([[-1.0]]))
    assert _q(env) < 0.0


def test_stop_loss_fires_early_despite_min_hold():
    # A crash with SL enabled must exit via stop_loss on the way down even
    # when the commitment floor would forbid a jitter close.
    closes, highs, lows = _crash_series()
    env = TradingEnv(
        mx.zeros((1, closes.size, 4)),
        mx.array(np.expand_dims(closes, axis=0)),
        highs=mx.array(np.expand_dims(highs, axis=0)),
        lows=mx.array(np.expand_dims(lows, axis=0)),
        n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
        lev_min=2.0, lev_max=20.0, risk_min=0.05, risk_max=0.05,
        take_profit=0.0, stop_loss=0.05,
        use_take_profit=False, use_stop_loss=True,
        trade_knob=1.0, open_fee_rate=0.0, close_fee_rate=0.0,
        holding_fee_daily=0.0, margin_mode="isolated", seed=1,
        min_hold_bars=999,
    )
    env.reset()
    exits = []
    for _ in range(10):
        obs, r, done, info = env.step(mx.array([[1.0]]))
        exits.extend(np.asarray(info["exit"]).astype(int).reshape(-1))
    assert 2 in exits  # stop_loss id fires despite the floor
