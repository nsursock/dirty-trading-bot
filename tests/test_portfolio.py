"""Portfolio-level evaluation tests.

Covers: multi-episode reporting (`eval.episodes`), the mean-of-accounts
portfolio identity (net == per-symbol average when one slot per symbol),
close-time trade sorting, and aggregate/per-episode figure emission.
"""

from pathlib import Path

import numpy as np
import pytest

from config import load

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke.yaml"


def _synthetic_episode(offs_i, n_sym=4, n_steps=8, seed_off=0, closed=-1):
    rng = np.random.default_rng(offs_i)
    # Each symbol account starts at 1000 and wanders independently.
    per_symbol = np.full((n_sym, n_steps), 1000.0)
    for s in range(n_sym):
        per_symbol[s] += np.cumsum(rng.normal(1e-2, 0.5, n_steps))
    net = per_symbol.mean(axis=0)
    trades = [{
        "trade_id": i, "symbol": f"SYM{s}", "side": "long",
        "opened_at": 1, "closed_at": closed if closed > 0 else rng.integers(2, n_steps),
        "entry_price": 100.0, "exit_price": 101.0, "notional": 100.0,
        "leverage": 1.0, "fee": 0.05, "realized_pnl": 1.0, "exit_type": "market_close",
    } for i, s in enumerate(range(n_sym))]
    return {
        "ledger": trades,
        "net": net,
        "gross": net + 0.5,
        "steps": np.arange(n_steps, dtype=float),
        "per_symbol": per_symbol,
        "sym_by_env": np.arange(n_sym),
        "seed_offset": seed_off,
    }


def test_episode_offsets_from_eval_config():
    from report import _episode_offsets, TEST_SEED_OFFSETS

    assert _episode_offsets(load(SMOKE)) == list(TEST_SEED_OFFSETS[:2])
    # Episodes beyond the locked draws extend the sequence deterministically
    # (offsets 9..N) and are never capped by ``len(TEST_SEED_OFFSETS)``.
    from config import Config

    n = len(TEST_SEED_OFFSETS) + 3
    assert _episode_offsets(Config({"eval": {"episodes": n}})) == list(range(1, n + 1))
    assert _episode_offsets(Config({"eval": {"episodes": 1}})) == [1]


def test_sort_ledger_by_close_time():
    from report import _sort_ledger, write_ledger

    trades = [
        {"trade_id": 0, "closed_at": 20, "symbol": "B", "episode": 0},
        {"trade_id": 1, "closed_at": 5, "symbol": "A", "episode": 1},
        {"trade_id": 2, "closed_at": 5, "symbol": "B", "episode": 0},
        {"trade_id": 3, "closed_at": None, "symbol": "A", "episode": 0},
    ]
    out = _sort_ledger(trades)
    assert [t["trade_id"] for t in out] == [1, 2, 0, 3]  # None-last

    path = REPO / "logs" / "test_trades.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_ledger(out, path)
    with open(path) as fh:
        header = fh.readline()
    assert "episode" in header
    path.unlink(missing_ok=True)


def test_aggregate_keeps_portfolio_mean_identity():
    from report import _aggregate_episodes

    ep1 = _synthetic_episode(1)
    ep2 = _synthetic_episode(2)
    agg = _aggregate_episodes([ep1, ep2])

    # Portfolio net is the mean of the account curves...
    assert np.allclose(agg["net"], (ep1["net"] + ep2["net"]) / 2)
    # ...which equals the average of per-symbol equity (one slot per symbol).
    assert np.allclose(agg["net"], agg["per_symbol"].mean(axis=0))
    # Combined ledger sorted by close time with episode tags.
    assert len(agg["ledger"]) == len(ep1["ledger"]) + len(ep2["ledger"])
    closes = [t["closed_at"] for t in agg["ledger"] if t["closed_at"] is not None]
    assert closes == sorted(closes)


def test_breakdown_includes_per_episode_section(tmp_path):
    from report import _aggregate_episodes, breakdown

    ep1, ep2 = _synthetic_episode(1, seed_off=1), _synthetic_episode(2, seed_off=2)
    for e in (ep1, ep2):
        for i, t in enumerate(e["ledger"]):
            t["episode"] = e["seed_offset"] - 1
            t["seed_offset"] = e["seed_offset"]
    agg = _aggregate_episodes([ep1, ep2])
    text = breakdown(agg, tmp_path / "breakdown.txt")
    assert "By episode" in text
    assert "episode 0 (seed+1)" in text
    assert "episode 1 (seed+2)" in text
    assert "all" in text


def test_aggregate_averages_roc_series():
    from report import _aggregate_episodes

    ep1 = _synthetic_episode(1, seed_off=1)
    ep2 = _synthetic_episode(2, seed_off=2)
    ep1["roc"] = np.linspace(0.0, 0.02, 8)
    ep2["roc"] = np.linspace(0.0, -0.01, 8)
    agg = _aggregate_episodes([ep1, ep2])
    assert np.allclose(agg["roc"], (ep1["roc"] + ep2["roc"]) / 2)


def test_metrics_accepts_collateral_return_series():
    from report import metrics

    net = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    rets = np.array([0.10, -0.05, 0.20])  # collateral-basis ROC series
    m = metrics(net, periods_per_year=72576.0, rets=rets, basis="collateral")
    assert m["return_basis"] == "collateral"
    assert m["sharpe"] == pytest.approx(rets.mean() / (rets.std() + 1e-12) * np.sqrt(72576.0), rel=1e-9)
    assert m["final_equity"] == pytest.approx(1003.0)  # dollar facts stay on equity


def test_figure1_renders_with_per_symbol_and_episode_overlays(tmp_path):
    from report import figure1

    ep = _synthetic_episode(3)
    result = {
        "ledger": ep["ledger"],
        "net": ep["net"],
        "gross": ep["gross"],
        "steps": ep["steps"],
        "per_symbol": ep["per_symbol"],
        "episodes": [{"net": ep["net"], "gross": ep["gross"], "steps": ep["steps"]}],
    }
    out = figure1(result, tmp_path / "figure1.png")
    assert out.exists() and out.stat().st_size > 0


def test_figure1_overlays_switch_disables_overlay_traces(tmp_path):
    from report import figure1

    ep = _synthetic_episode(4)
    result = {
        "ledger": [],
        "net": ep["net"],
        "gross": ep["gross"],
        "steps": ep["steps"],
        "per_symbol": ep["per_symbol"],
        "episodes": [{"net": ep["net"], "gross": ep["gross"], "steps": ep["steps"]}],
    }
    out = figure1(result, tmp_path / "plain.png", overlays=False)
    assert out.exists() and out.stat().st_size > 0