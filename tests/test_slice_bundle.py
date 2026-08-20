"""Tests for causal bundle slicing."""

from __future__ import annotations

import mlx.core as mx

from data import generate, slice_bundle


def test_slice_bundle_shapes_and_bounds():
    bundle = generate(
        symbols={"BTC": (0.0, 0.6, 60_000.0), "ETH": (0.0, 0.7, 3_000.0)},
        n_steps=240,
        seed=7,
        low_tf="5m",
        high_tf="4h",
        regime="neutral",
        ar_coef=0.35,
        ar_noise=1.71,
    )
    sl = slice_bundle(bundle, 40, 160)
    assert sl.features.shape[1] == 120
    assert sl.ohlcv.closes.shape == (2, 120)
    assert sl.n_resample == bundle.n_resample
    # High-TF length follows low-TF floor division.
    assert sl.high_features.shape[1] == 120 // sl.n_resample


def test_slice_bundle_rejects_bad_range():
    bundle = generate(
        symbols={"BTC": (0.0, 0.6, 60_000.0)},
        n_steps=100,
        seed=1,
        low_tf="5m",
        high_tf="4h",
        regime="neutral",
    )
    try:
        slice_bundle(bundle, 50, 50)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        slice_bundle(bundle, -1, 10)
        assert False, "expected ValueError"
    except ValueError:
        pass
    mx.eval(bundle.features)  # keep mlx graph tidy in test process
