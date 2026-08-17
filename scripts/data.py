"""Synthetic GBM OHLCV + pure-MLX TA feature tensor.

Generates a 5-10 symbol crypto universe (each symbol its own annual drift
``mu`` and vol ``sigma``) through ``dirty-mkt-data`` (Apple MLX only), builds
OHLCV bars, computes a vectorized TA feature stack, and returns an MLX tensor
``(n_symbols, n_steps, n_features)`` for the vectorized env.

Exponential-moving-average indicators (EMA-ratio, MACD, RSI, ATR, volume
surprise) use an IIR recursion that MLX has no ``scan`` for: each is a short
loop over time, vectorized over the symbol axis. All other kernels are single
cumsum / rolling ops over ``(n_symbols, n_steps)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx
from dirty_mkt_data import Generator
from dirty_mkt_data.core.gbm import GBM
from dirty_mkt_data.eval.rolling import rolling_mean
from dirty_mkt_data.viz.ohlcv import OHLCV, from_dataset
from tqdm import tqdm

log = logging.getLogger("trading")

TRADING_DAYS = 252
EPS = 1e-12

SYMBOLS: dict[str, tuple[float, float, float]] = {
    "BTC": (0.30, 0.60, 60_000.0),
    "ETH": (0.35, 0.75, 3_000.0),
    "SOL": (0.50, 1.00, 150.0),
    "BNB": (0.25, 0.55, 550.0),
    "XRP": (0.20, 0.80, 0.60),
    "DOGE": (0.15, 1.10, 0.15),
    "AVAX": (0.40, 0.90, 35.0),
    "LINK": (0.30, 0.70, 15.0),
}

FEATURES: tuple[str, ...] = (
    "ret1",
    "ret5",
    "ret10",
    "ret20",
    "ema_ratio",
    "macd_hist",
    "rsi",
    "atr_pct",
    "rv10",
    "rv20",
    "vol_z",
    "vol_surprise",
    "range_pct",
    "body_pct",
    "close_loc",
)


@dataclass(frozen=True)
class DataBundle:
    symbols: tuple[str, ...]
    ohlcv: OHLCV
    features: mx.array
    feature_names: tuple[str, ...] = FEATURES


def _lag(x: mx.array, k: int) -> mx.array:
    if k <= 0:
        return x
    head = mx.broadcast_to(x[:, :1], (x.shape[0], k))
    return mx.concatenate([head, x[:, :-k]], axis=1)


def _ewm(x: mx.array, alpha: float) -> mx.array:
    cols = [x[:, 0]]
    y = x[:, 0]
    for t in range(1, x.shape[1]):
        y = alpha * x[:, t] + (1.0 - alpha) * y
        cols.append(y)
    return mx.stack(cols, axis=1)


def _rolling_std(x: mx.array, window: int) -> mx.array:
    m = rolling_mean(x, window)
    m2 = rolling_mean(x * x, window)
    return mx.sqrt(mx.maximum(m2 - m * m, 0.0))


def _rsi(closes: mx.array, window: int = 14) -> mx.array:
    delta = closes - _lag(closes, 1)
    gain = _ewm(mx.maximum(delta, 0.0), 1.0 / window)
    loss = _ewm(mx.maximum(-delta, 0.0), 1.0 / window)
    return 100.0 - 100.0 / (1.0 + gain / (loss + EPS))


def _atr(highs: mx.array, lows: mx.array, closes: mx.array, window: int = 14) -> mx.array:
    pc = _lag(closes, 1)
    tr = mx.maximum(highs - lows, mx.maximum(mx.abs(highs - pc), mx.abs(lows - pc)))
    return _ewm(tr, 1.0 / window)


def _macd(closes: mx.array, fast: int = 12, slow: int = 26, signal: int = 9) -> mx.array:
    line = _ewm(closes, 2.0 / (fast + 1)) - _ewm(closes, 2.0 / (slow + 1))
    sig = _ewm(line, 2.0 / (signal + 1))
    return line - sig


def _features(o: OHLCV) -> mx.array:
    c, h, l, v = o.closes, o.highs, o.lows, o.vols
    logc = mx.log(c)
    r1 = logc - _lag(logc, 1)
    ema20 = _ewm(c, 2.0 / 21.0)
    ema_v = _ewm(v, 2.0 / 21.0)
    return mx.stack(
        [
            r1,
            logc - _lag(logc, 5),
            logc - _lag(logc, 10),
            logc - _lag(logc, 20),
            c / (ema20 + EPS) - 1.0,
            _macd(c) / (c + EPS),
            (_rsi(c) - 50.0) / 50.0,
            _atr(h, l, c) / (c + EPS),
            _rolling_std(r1, 10),
            _rolling_std(r1, 20),
            (v - rolling_mean(v, 20)) / (_rolling_std(v, 20) + EPS),
            v / (ema_v + EPS) - 1.0,
            (h - l) / (c + EPS),
            mx.abs(c - o.opens) / (c + EPS),
            (c - l) / (h - l + EPS) - 0.5,
        ],
        axis=-1,
    )


def generate(
    symbols: dict[str, tuple[float, float, float]] | None = None,
    n_steps: int = 2520,
    seed: int = 42,
    dt: float = 1.0 / TRADING_DAYS,
    base_volume: float = 1_000_000.0,
) -> DataBundle:
    params = SYMBOLS if symbols is None else symbols
    names = tuple(params)
    keys = mx.random.split(mx.random.key(seed), len(names))
    log.info("data: generating %d symbols x %d steps (seed=%d, dt=%.6f)", len(names), n_steps, seed, dt)

    arrays = {k: [] for k in ("opens", "highs", "lows", "closes", "vols")}
    for i, name in enumerate(tqdm(names, desc="symbols", leave=False)):
        mu, sigma, s0 = params[name]
        log.debug("data: symbol=%s mu=%.4f sigma=%.4f s0=%.4f", name, mu, sigma, s0)
        ds = Generator(GBM(mu=mu, sigma=sigma, s0=s0, dt=dt), seed=seed).sample(
            n_steps, n_paths=1, run_id=i
        )
        o = from_dataset(ds, sigma=sigma, dt=dt, base_volume=base_volume, key=keys[i])
        arrays["opens"].append(o.opens)
        arrays["highs"].append(o.highs)
        arrays["lows"].append(o.lows)
        arrays["closes"].append(o.closes)
        arrays["vols"].append(o.vols)

    ohlcv = OHLCV(
        opens=mx.concatenate(arrays["opens"], axis=0),
        highs=mx.concatenate(arrays["highs"], axis=0),
        lows=mx.concatenate(arrays["lows"], axis=0),
        closes=mx.concatenate(arrays["closes"], axis=0),
        vols=mx.concatenate(arrays["vols"], axis=0),
    )
    feats = _features(ohlcv)
    log.debug("data: feature tensor %s (%d features)", feats.shape, len(FEATURES))
    return DataBundle(symbols=names, ohlcv=ohlcv, features=feats)
