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
import time
from dataclasses import dataclass

import mlx.core as mx
from dirty_mkt_data import Generator
from dirty_mkt_data.core.argbm import ARGBM
from dirty_mkt_data.eval.rolling import rolling_mean
from dirty_mkt_data.viz.ohlcv import OHLCV, from_dataset
from tqdm import tqdm

log = logging.getLogger("trading")

TRADING_DAYS = 252
EPS = 1e-12

# Binance klines v3 intervals (as returned by GET /api/v3/klines).
BINANCE_TFS: frozenset[str] = frozenset(
    {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d",
        "1w",
        "1M",
    }
)


def _tf_minutes(tf: int | str) -> int:
    """Convert a Binance klines v3 interval like ``"4h"`` to minutes.

    Integers are accepted as minutes for backward compatibility.
    """
    if isinstance(tf, int):
        return tf
    if not isinstance(tf, str) or tf not in BINANCE_TFS:
        raise ValueError(f"unknown Binance timeframe: {tf!r} (want one of {sorted(BINANCE_TFS)})")
    v, unit = int(tf[:-1]), tf[-1]
    scale = {"m": 1, "h": 60, "d": 60 * 24, "w": 60 * 24 * 7, "M": 60 * 24 * 30}
    return v * scale[unit]

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
    high_ohlcv: OHLCV | None = None
    high_features: mx.array | None = None
    n_resample: int = 1


def _resample(o: OHLCV, n: int) -> OHLCV:
    """Aggregate every ``n`` low-TF bars into one high-TF bar (open/high/low/close/vol).

    Causality rule: high-TF window ``i`` spans low-TF bars ``[i*n, (i+1)*n)``
    and only closes once ``low_steps >= (i+1)*n``. Consumers must read window
    ``i`` strictly after it closes --- see ``build_high_view`` / ``mgr_obs``.
    """
    S, T = o.closes.shape
    T_hi = T // n
    if T_hi == 0:
        raise ValueError(f"n_steps={T} too small for resample n={n}")
    sl = T_hi * n
    x = mx.reshape(o.closes[:, :sl], (S, T_hi, n))
    return OHLCV(
        opens=mx.reshape(o.opens[:, :sl], (S, T_hi, n))[:, :, 0],
        highs=mx.max(mx.reshape(o.highs[:, :sl], (S, T_hi, n)), axis=-1),
        lows=mx.min(mx.reshape(o.lows[:, :sl], (S, T_hi, n)), axis=-1),
        closes=x[:, :, -1],
        vols=mx.sum(mx.reshape(o.vols[:, :sl], (S, T_hi, n)), axis=-1),
    )


def build_high_view(high_features: mx.array) -> mx.array:
    """Shift high-TF features so that row ``j`` describes window ``j-1``.

    Row ``0`` is a pre-launch placeholder with no dependency on any low-TF
    bar; row ``j >= 1`` holds the features of the already-closed high-TF
    window ``j-1``. Combined with ``mgr_obs``' ``low_steps // n_resample``
    completion index, consumers only ever read windows strictly before the
    bar that is still forming (causal, no lookahead).
    """
    S, T_hi, F = high_features.shape
    shifted = mx.concatenate([mx.zeros((S, 1, F)), high_features[:, :-1, :]], axis=1)
    return mx.reshape(shifted, (S * T_hi, F))


def mgr_obs(feats_hi, acct, low_steps, sym_off, T_hi, n_resample):
    """Manager high-TF observation: last *completed* window + account state.

    ``feats_hi`` is the 2-D view from ``build_high_view``; ``low_steps`` is
    the env step index. Only windows that closed at or before
    ``low_steps // n_resample`` are visible.
    """
    idx = mx.minimum(low_steps // n_resample, T_hi - 1)
    feats = mx.take(feats_hi, sym_off + idx, axis=0)
    return mx.concatenate([feats, acct], axis=1)


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
    T = c.shape[1]

    def lag(x, k):
        k = max(min(k, max(T - 1, 1)), 1)
        return _lag(x, k)

    ema20 = _ewm(c, 2.0 / 21.0)
    ema_v = _ewm(v, 2.0 / 21.0)
    return mx.stack(
        [
            r1,
            logc - lag(logc, 5),
            logc - lag(logc, 10),
            logc - lag(logc, 20),
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
    low_tf: int | str = "5m",
    high_tf: int | str = "4h",
    base_volume: float = 1_000_000.0,
    regime: str = "bull",
    dt: float | None = None,
    ar_coef: float = 0.0,
) -> DataBundle:
    params = SYMBOLS if symbols is None else symbols
    names = tuple(params)
    keys = mx.random.split(mx.random.key(seed), len(names))
    low_min = _tf_minutes(low_tf)
    high_min = _tf_minutes(high_tf)
    if dt is None:
        dt = low_min / (60 * 24 * TRADING_DAYS)
    n = max(high_min // low_min, 1)

    mus = [params[s][0] for s in names]
    sigmas = [params[s][1] for s in names]
    s0s = [params[s][2] for s in names]
    if regime == "neutral":
        mus = [0.0] * len(names)
    elif regime == "mix":
        for i in range(len(names)):
            mus[i] = mus[i] if i % 3 == 0 else (-mus[i] if i % 3 == 1 else 0.0)
    elif regime != "bull":
        raise ValueError(f"unknown regime: {regime}")
    params = {s: (mus[i], sigmas[i], s0s[i]) for i, s in enumerate(names)}

    log.info(
        "data: generating %d symbols x %d low-TF(%sm) steps, high-TF(%sm) x%d, regime=%s (seed=%d, dt=%.2e)",
        len(names), n_steps, low_min, high_min, n, regime, seed, dt,
    )
    if ar_coef:
        log.info("data: injecting AR(1) alpha phi=%.2f into GBM log-returns", ar_coef)

    arrays = {k: [] for k in ("opens", "highs", "lows", "closes", "vols")}
    t_gen = time.monotonic()
    for i, name in enumerate(tqdm(names, desc="symbols", leave=False)):
        t0 = time.monotonic()
        mu, sigma, s0 = params[name]
        log.debug("data: symbol=%s mu=%.4f sigma=%.4f s0=%.4f", name, mu, sigma, s0)
        # ARGBM(phi=0) is exactly GBM (drift + iid noise), so the single model
        # covers the null and every injected-alpha strength.
        model = ARGBM(mu=mu, sigma=sigma, s0=s0, dt=dt, phi=ar_coef)
        ds = Generator(model, seed=seed).sample(
            n_steps, n_paths=1, run_id=i
        )
        o = from_dataset(ds, sigma=sigma, dt=dt, base_volume=base_volume, key=keys[i])
        arrays["opens"].append(o.opens)
        arrays["highs"].append(o.highs)
        arrays["lows"].append(o.lows)
        arrays["closes"].append(o.closes)
        arrays["vols"].append(o.vols)
        log.debug("data: symbol=%s GBM+OHLCV in %.2fs", name, time.monotonic() - t0)

    log.info("data: %d symbols x %d low-TF steps generated in %.2fs",
             len(names), n_steps, time.monotonic() - t_gen)

    ohlcv = OHLCV(
        opens=mx.concatenate(arrays["opens"], axis=0),
        highs=mx.concatenate(arrays["highs"], axis=0),
        lows=mx.concatenate(arrays["lows"], axis=0),
        closes=mx.concatenate(arrays["closes"], axis=0),
        vols=mx.concatenate(arrays["vols"], axis=0),
    )
    feats = _features(ohlcv)
    high_ohlcv = _resample(ohlcv, n) if n > 1 else ohlcv
    high_features = _features(high_ohlcv) if n > 1 else feats
    log.debug("data: low features %s, high features %s (n_resample=%d)", feats.shape, high_features.shape, n)
    return DataBundle(
        symbols=names, ohlcv=ohlcv, features=feats,
        high_ohlcv=high_ohlcv, high_features=high_features, n_resample=n,
    )
