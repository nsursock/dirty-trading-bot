"""TP / SL switch tests.

Verifies use_take_profit / use_stop_loss independently gate the two
intrabar exits, while liquidations always remain active.
"""

import mlx.core as mx
import numpy as np
import pytest
from env import TradingEnv


def _force_long_series():
    """A monotone-up then sharp-down series that would trip both TP and SL."""
    closes = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0,
                       108.0, 102.0, 96.0, 90.0, 84.0, 78.0], dtype=np.float32)
    highs = np.maximum(closes, np.roll(closes, 1) * 1.01).astype(np.float32)
    lows = np.minimum(closes, np.roll(closes, 1) * 0.99).astype(np.float32)
    return closes, highs, lows


def _make_env(**over):
    closes, highs, lows = _force_long_series()
    kw = dict(
        n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
        lev_min=2.0, lev_max=20.0, risk_min=0.05, risk_max=0.05,
        take_profit=0.05, stop_loss=0.05, trade_knob=1.0,
        open_fee_rate=0.0, close_fee_rate=0.0, holding_fee_daily=0.0,
        margin_mode="isolated", seed=1,
    )
    kw.update(over)
    feats = mx.zeros((1, closes.size, 4))
    return TradingEnv(feats,
                      mx.array(np.expand_dims(closes, axis=0)),
                      highs=mx.array(np.expand_dims(highs, axis=0)),
                      lows=mx.array(np.expand_dims(lows, axis=0)), **kw)


def _run(env, n=6):
    env.reset()
    exits = []
    for _ in range(n):
        obs, r, done, info = env.step(mx.array([[1.0]]))
        exits.extend(np.asarray(info["exit"]).astype(int).reshape(-1))
    return np.asarray(exits)


def test_tp_and_sl_both_off_never_intrabar_exit():
    env = _make_env(use_take_profit=False, use_stop_loss=False)
    exits = _run(env)
    assert (exits == 0).all()  # only market_close would come from the policy


def test_tp_off_sl_on_still_gives_sl_exits():
    env = _make_env(use_take_profit=False, use_stop_loss=True)
    exits = _run(env, n=10)
    # Sharp drop partway in should hit stop_loss.
    assert 2 in exits  # stop_loss id


def test_sl_off_tp_on_still_gives_tp_exits():
    env = _make_env(use_take_profit=True, use_stop_loss=False)
    exits = _run(env, n=10)
    # Round trip above TP first before the drop; with SL off the ride-down
    # later liquidates, but a take_profit exit must still have fired.
    assert 1 in exits  # take_profit id


def test_liquidation_never_disabled():
    # A crashing series at 150x with both switches off must still liquidate.
    closes = np.array([100.0, 99.0, 97.0, 94.0, 90.0, 86.0,
                       81.0, 76.0, 70.0, 64.0, 58.0, 52.0], dtype=np.float32)
    highs = np.maximum(closes, np.roll(closes, 1) * 1.01).astype(np.float32)
    lows = np.minimum(closes, np.roll(closes, 1) * 0.99).astype(np.float32)
    env = TradingEnv(
        mx.zeros((1, closes.size, 4)),
        mx.array(np.expand_dims(closes, axis=0)),
        highs=mx.array(np.expand_dims(highs, axis=0)),
        lows=mx.array(np.expand_dims(lows, axis=0)),
        n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
        lev_min=150.0, lev_max=150.0, risk_min=0.1, risk_max=0.1,
        take_profit=0.05, stop_loss=0.05,
        use_take_profit=False, use_stop_loss=False,
        trade_knob=1.0, open_fee_rate=0.0, close_fee_rate=0.0,
        holding_fee_daily=0.0, margin_mode="isolated", seed=1,
    )
    env.reset()
    exits = []
    for _ in range(11):
        obs, r, done, info = env.step(mx.array([[1.0]]))
        exits.extend(np.asarray(info["exit"]).astype(int).reshape(-1))
    assert 3 in exits  # liquidation fires even with TP/SL switched off


def test_isolated_liquidation_does_not_reset_account():
    """Ref #9: liq closes the position; episode continues with residual cash."""
    closes = np.array([100.0, 99.0, 97.0, 94.0, 90.0, 86.0,
                       81.0, 76.0, 70.0, 64.0, 58.0, 52.0], dtype=np.float32)
    highs = np.maximum(closes, np.roll(closes, 1) * 1.01).astype(np.float32)
    lows = np.minimum(closes, np.roll(closes, 1) * 0.99).astype(np.float32)
    env = TradingEnv(
        mx.zeros((1, closes.size, 4)),
        mx.array(np.expand_dims(closes, axis=0)),
        highs=mx.array(np.expand_dims(highs, axis=0)),
        lows=mx.array(np.expand_dims(lows, axis=0)),
        n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
        lev_min=150.0, lev_max=150.0, risk_min=0.1, risk_max=0.1,
        take_profit=0.05, stop_loss=0.05,
        use_take_profit=False, use_stop_loss=False,
        trade_knob=1.0, open_fee_rate=0.0, close_fee_rate=0.0,
        holding_fee_daily=0.0, margin_mode="isolated",
        min_collateral=1.0, seed=1,
    )
    env.reset()
    liq_step = None
    bal_after_liq = None
    for i in range(11):
        obs, r, done, info = env.step(mx.array([[1.0]]))
        exit_flag = int(np.asarray(info["exit"]).reshape(-1)[0])
        state = np.asarray(env._state)
        bal, qty, coll = float(state[0, 0]), float(state[0, 1]), float(state[0, 3])
        if exit_flag == 3 and liq_step is None:
            liq_step = i
            bal_after_liq = bal
            # Isolated liq must clear the position without ending the episode
            # unless equity is already below min_collateral.
            assert abs(qty) < 1e-6
            assert abs(coll) < 1e-6
            assert not bool(np.asarray(done).reshape(-1)[0])
            assert bal_after_liq < 1000.0
            assert bal_after_liq != pytest.approx(1000.0)
        elif liq_step is not None and i == liq_step + 1:
            # Next step must NOT auto-reset cash back to initial_balance.
            assert bal != pytest.approx(1000.0)
            assert bal == pytest.approx(bal_after_liq, abs=1.0) or bal < 1000.0
    assert liq_step is not None, "expected a liquidation on the crash series"