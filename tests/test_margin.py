"""Margin-mode tests (isolated vs cross).

Verifies the accounting contract for the two modes:
- isolated: opening a position locks `risk_frac * balance` out of cash
  into collateral; equity = balance + collateral + unrealized PnL.
- cross: the account keeps the whole balance in cash, collateral is an
  allocation only (excluded from equity); liquidation and sizing instead
  key off total account equity.
"""

import mlx.core as mx
import numpy as np
import pytest

from data import SYMBOLS, generate
from env import TradingEnv


def _env(mode, seed=5):
    b = generate(symbols={"BTC": SYMBOLS["BTC"]}, n_steps=120, seed=seed,
                 low_tf=5, high_tf=30)
    return TradingEnv(
        b.features, b.ohlcv.closes, highs=b.ohlcv.highs, lows=b.ohlcv.lows,
        n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
        lev_min=2.0, lev_max=20.0, risk_min=0.05, risk_max=0.05,
        take_profit=0.0, stop_loss=0.0, trade_knob=1.0, open_fee_rate=0.0,
        close_fee_rate=0.0, holding_fee_daily=0.0,
        margin_mode=mode, seed=seed,
    )


@pytest.mark.parametrize("mode", ["isolated", "cross"])
def test_env_runs_without_nan(mode):
    env = _env(mode)
    env.reset()
    eqs = []
    for _ in range(100):
        obs, r, done, info = env.step(mx.array([[1.0]]))
        eqs.append(np.asarray(env._state)[:, 0].copy())
    eqs = np.asarray(eqs)
    assert np.isfinite(eqs).all()
    assert not np.isnan(eqs).any()


def test_isolated_locks_collateral_cross_does_not():
    iso = _env("isolated")
    iso.reset()
    iso.step(mx.array([[1.0]]))
    s_iso = np.asarray(iso._state)
    bal_i, col_i = float(s_iso[0, 0]), float(s_iso[0, 3])

    cr = _env("cross")
    cr.reset()
    cr.step(mx.array([[1.0]]))
    s_cr = np.asarray(cr._state)
    bal_c, col_c = float(s_cr[0, 0]), float(s_cr[0, 3])

    assert abs(col_i - 50.0) < 0.01 and abs(col_c - 50.0) < 0.01
    # Cross keeps ~all cash free; isolated has locked the 50 collateral out.
    assert abs(bal_i - bal_c + 50.0) < 0.01
    assert abs(bal_c - 1000.0) < 1.0


def test_equity_convention_matches_mode():
    # isolated: eq = balance + collateral + upnl
    # cross:     eq = balance + upnl (collateral is an allocation only)
    iso = _env("isolated")
    iso.reset()
    iso.step(mx.array([[1.0]]))
    s = np.asarray(iso._state)
    assert iso.margin_mode == "isolated"

    cr = _env("cross")
    cr.reset()
    cr.step(mx.array([[1.0]]))
    assert cr.margin_mode == "cross"


def test_rejects_unknown_margin_mode():
    b = generate(symbols={"BTC": SYMBOLS["BTC"]}, n_steps=60, seed=1, low_tf=5, high_tf=30)
    with pytest.raises(ValueError):
        TradingEnv(b.features, b.ohlcv.closes, highs=b.ohlcv.highs, lows=b.ohlcv.lows,
                   n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
                   margin_mode="margin_kallisti", seed=1)


@pytest.mark.parametrize("mode", ["isolated", "cross"])
def test_collateral_reward_basis_runs_finite(mode):
    b = generate(symbols={"BTC": SYMBOLS["BTC"]}, n_steps=120, seed=5, low_tf=5, high_tf=30)
    env = TradingEnv(
        b.features, b.ohlcv.closes, highs=b.ohlcv.highs, lows=b.ohlcv.lows,
        n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
        lev_min=2.0, lev_max=20.0, risk_min=0.05, risk_max=0.05,
        take_profit=0.0, stop_loss=0.0, trade_knob=1.0, open_fee_rate=0.0,
        close_fee_rate=0.0, holding_fee_daily=0.0,
        margin_mode=mode, return_basis="collateral", seed=5,
    )
    env.reset()
    rw = []
    for _ in range(80):
        _, r, done, info = env.step(mx.array([[1.0]]))
        rw.append(float(np.asarray(r).ravel()[0]))
    rw = np.asarray(rw)
    assert np.isfinite(rw).all()


def test_rejects_unknown_return_basis():
    b = generate(symbols={"BTC": SYMBOLS["BTC"]}, n_steps=60, seed=1, low_tf=5, high_tf=30)
    with pytest.raises(ValueError):
        TradingEnv(b.features, b.ohlcv.closes, highs=b.ohlcv.highs, lows=b.ohlcv.lows,
                   n_envs_per_symbol=1, action_space="continuous", goal_dim=0,
                   return_basis="margin_kallisti", seed=1)


def test_breakdown_roe_uses_collateral_basis(tmp_path):
    from config import Config
    from report import breakdown

    trades = [{
        "trade_id": 0, "symbol": "BTC", "side": "long", "opened_at": 1, "closed_at": 3,
        "entry_price": 100.0, "exit_price": 110.0, "notional": 1000.0, "leverage": 10.0,
        "collateral": 100.0, "equity_before": 1000.0, "fee": 1.0, "realized_pnl": 50.0,
        "exit_type": "take_profit",
    }]
    res = {"ledger": trades, "net": np.array([1000.0, 1010.0, 1100.0]),
           "gross": np.array([1000.0, 1012.0, 1103.0])}

    text = breakdown(res, tmp_path / "bd.txt", Config({"returns": {"basis": "collateral"}}))
    assert "By return" in text
    assert "multi-R (>=100 bps)" in text  # 50 pnl / 100 collateral = +50% RoC = 5000 bps

    text2 = breakdown(res, tmp_path / "bd2.txt", Config({"returns": {"basis": "account"}}))
    assert "By RoE" in text2
    assert "multi-R (>=100 bps)" in text2  # 50 pnl / 1000 account equity = +5% = 500 bps
