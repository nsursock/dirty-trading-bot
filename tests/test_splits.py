"""Unit tests for causal walk-forward / purged splits (Stage 0)."""

from __future__ import annotations

import pytest

from splits import assert_no_leakage, walk_forward_splits


def test_expanding_walk_forward_is_causal():
    folds = walk_forward_splits(
        800,
        n_folds=3,
        train_bars=300,
        test_bars=100,
        purge_bars=24,
        embargo_bars=24,
        mode="expanding",
    )
    assert len(folds) == 3
    assert_no_leakage(folds)
    assert folds[0].train_start == 0
    assert folds[0].train_end == 300
    assert folds[0].test_start == 348
    assert folds[0].test_end == 448
    # Expanding: later folds keep a longer train window.
    assert folds[1].train_start == 0
    assert folds[1].train_end == 400
    assert folds[1].test_start == 448
    assert folds[2].train_end == 500
    assert folds[2].test_end == 648


def test_rolling_walk_forward_keeps_train_length():
    folds = walk_forward_splits(
        800,
        n_folds=3,
        train_bars=300,
        test_bars=100,
        purge_bars=24,
        embargo_bars=24,
        mode="rolling",
    )
    assert_no_leakage(folds)
    for f in folds:
        assert f.train_bars == 300
        assert f.test_bars == 100
        assert f.train_end + 24 + 24 == f.test_start


def test_too_short_path_raises():
    with pytest.raises(ValueError, match="too short"):
        walk_forward_splits(
            400,
            n_folds=3,
            train_bars=300,
            test_bars=100,
            purge_bars=24,
            embargo_bars=24,
        )


def test_train_test_disjoint_sets():
    folds = walk_forward_splits(
        1000,
        n_folds=4,
        train_bars=200,
        test_bars=80,
        purge_bars=10,
        embargo_bars=10,
        mode="expanding",
    )
    for f in folds:
        assert set(range(f.train_start, f.train_end)).isdisjoint(
            range(f.test_start, f.test_end)
        )
