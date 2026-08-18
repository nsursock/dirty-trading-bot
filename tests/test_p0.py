"""P0 regression tests for the 260818 audit action list.

Covers: metric identity (#1), Optuna/validation split (#2), checkpoint
manifest binding (#3), and manager high-TF lookahead (#4).
"""

from pathlib import Path

import numpy as np
import pytest

from config import load

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke.yaml"
pytestmark = pytest.mark.p0


# --- #1: metric identity ---------------------------------------------------


def test_metrics_matches_hand_computed():
    from report import _returns, metrics

    ppy = 72576.0  # 5-minute bars: 252 * 1440 / 5
    net = np.array([1000.0, 1020.0, 1010.0, 1030.0, 1020.0, 1040.0])
    rets = _returns(net)
    mean, std = rets.mean(), rets.std(ddof=0)
    down = rets[rets < 0]
    years = len(net) / ppy
    m = metrics(net, periods_per_year=ppy)
    assert m["sharpe"] == pytest.approx(mean / std * np.sqrt(ppy), rel=1e-9)
    assert m["sortino"] == pytest.approx(mean / (down.std(ddof=0) + 1e-12) * np.sqrt(ppy), rel=1e-9)
    assert m["cagr"] == pytest.approx((net[-1] / net[0]) ** (1.0 / years) - 1.0, rel=1e-9)
    assert m["total_return"] == pytest.approx(net[-1] / net[0] - 1.0)
    assert m["final_equity"] == pytest.approx(net[-1])


def test_periods_per_year_derived_from_bar_not_hardcoded_252():
    from report import _periods_per_year

    cfg = load(SMOKE)
    assert _periods_per_year(cfg) == pytest.approx(252 * 1440 / 5)  # 72576, not 252


def test_trade_stats_not_annualized():
    from report import _trade_stats

    st = _trade_stats([{"realized_pnl": 10.0}, {"realized_pnl": -5.0}, {"realized_pnl": 2.0}])
    assert st["num"] == 3
    assert st["net"] == pytest.approx(7.0)
    # no fake sqrt(n) annualization on per-trade PnL
    assert st["sharpe"] == 0.0
    assert st["sortino"] == 0.0
    assert st["calmar"] == 0.0


# --- #2: Optuna objective split from locked test ---------------------------


def test_validation_and_test_bundles_disjoint():
    from report import TEST_SEED_OFFSETS, VALID_SEED_OFFSETS

    assert len(VALID_SEED_OFFSETS) >= 2
    assert not (set(TEST_SEED_OFFSETS) & set(VALID_SEED_OFFSETS))


def test_dsr_deflates_with_more_trials():
    from report import dsr

    rng = np.random.default_rng(0)
    net = 1000.0 + np.cumsum(rng.normal(1e-4, 2e-3, 600))
    d1 = dsr(net, periods_per_year=72576, n_trials=1)
    d200 = dsr(net, periods_per_year=72576, n_trials=200)
    assert d200["dsr_probability"] <= d1["dsr_probability"] + 1e-9
    assert d200["deflated_sharpe"] <= d1["deflated_sharpe"] + 1e-9


# --- #3: checkpoint manifest binding ---------------------------------------


def test_config_hash_stable_and_change_sensitive():
    from agents import config_hash

    assert config_hash(load(SMOKE)) == config_hash(load(SMOKE))
    cfg = load(SMOKE)
    cfg["seed"] = 7
    assert config_hash(load(SMOKE)) != config_hash(cfg)


# --- #4: manager high-TF lookahead -----------------------------------------


def _synth_ohlcv(closes):
    S, T = closes.shape
    opens = closes - 0.5
    highs = closes + 0.5
    lows = closes - 0.5
    vols = mx.broadcast_to(mx.arange(T, dtype=mx.float32) * 10.0 + 5.0, (S, T))
    from dirty_mkt_data.viz.ohlcv import OHLCV

    return OHLCV(opens=opens, highs=highs, lows=lows, closes=closes, vols=vols)


def test_mgr_obs_never_reads_a_forming_window():
    """Row ``j`` of the manager view must resolve to already-closed window
    ``j-1`` (or the pre-launch placeholder in the first window)."""
    from data import build_high_view, mgr_obs

    n, S, T_hi, F = 4, 1, 8, 4
    feats = mx.broadcast_to(
        (mx.arange(T_hi, dtype=mx.float32) + 1.0)[None, :, None], (S, T_hi, F)
    )
    view = build_high_view(feats)
    sym_off = mx.array([0], dtype=mx.int32)
    acct = mx.zeros((1, F), mx.float32)

    for k in [0, n - 1, n, 2 * n - 1, 2 * n, 3 * n - 1]:
        low_steps = mx.array([k], dtype=mx.int32)
        obs = mgr_obs(view, acct, low_steps, sym_off, T_hi, n)
        got = float(mx.mean(obs[0, :F]).item())
        if k < n:
            assert got == pytest.approx(0.0)  # pre-launch placeholder, no leak
        else:
            # view row k//n == high_features[k//n - 1] => value (k//n - 1) + 1
            assert got == pytest.approx(float(k // n))


def test_resample_windows_close_forward_only():
    """Perturbing low-TF bars in a later window must not change earlier
    resampled windows (causal boundary of ``_resample``)."""
    from data import _resample

    S, T, n = 1, 16, 4
    base = mx.arange(T, dtype=mx.float32) + 100.0
    o = _synth_ohlcv(mx.broadcast_to(base, (S, T)))
    hi = _resample(o, n)
    closes2 = mx.concatenate([base[:12], base[12:] + 200.0])  # perturb last window [12, 16)
    o2 = _synth_ohlcv(mx.broadcast_to(closes2, (S, T)))
    hi2 = _resample(o2, n)

    assert mx.allclose(hi.closes[:, :3], hi2.closes[:, :3]).item()
    assert not mx.allclose(hi.closes[:, 3], hi2.closes[:, 3]).item()


def test_mgr_features_at_step_k_ignore_low_bars_k_plus():
    """HTF features observed by the manager at step k are unchanged when the
    low-TF bars k+1.. are perturbed."""
    from data import _features, _resample, build_high_view, mgr_obs

    S, T, n = 1, 48, 4
    base = mx.arange(T, dtype=mx.float32) + 100.0
    hi_a = _features(_resample(_synth_ohlcv(mx.broadcast_to(base, (S, T))), n))
    T_hi, F = hi_a.shape[1], hi_a.shape[2]
    view_a = build_high_view(hi_a)
    sym_off = mx.array([0], dtype=mx.int32)
    acct = mx.zeros((1, 6), mx.float32)

    for k in [1, 7, 13, 21, 27]:
        perturb = mx.where(mx.arange(T)[None, :] > k, 500.0, 0.0)
        closes_b = base[None, :] + perturb
        hi_b = _features(_resample(_synth_ohlcv(closes_b), n))
        view_b = build_high_view(hi_b)
        low_steps = mx.array([k], dtype=mx.int32)
        obs_a = mgr_obs(view_a, acct, low_steps, sym_off, T_hi, n)
        obs_b = mgr_obs(view_b, acct, low_steps, sym_off, T_hi, n)
        assert mx.allclose(obs_a[:, :F], obs_b[:, :F]).item(), f"lookahead leak at k={k}"