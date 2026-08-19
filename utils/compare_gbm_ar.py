"""Side-by-side candlesticks: pure GBM vs AR(1)-injected GBM (same seed).

Renders two PNGs — the pure GBM series (phi=0) and the AR-injected series
(default phi=0.7) — both from the identical first ``--bars`` low-TF bars of
the same bundle, so the drift structure differs only by the injected
momentum autocorrelation.

    venv/bin/python utils/compare_gbm_ar.py --bars 100 --phi 0.7 --out figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx  # noqa: F401  (ensure mlx is importable before dirty_mkt_data)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data import generate  # noqa: E402
from dirty_mkt_data.viz.inspect import candle_figure  # noqa: E402
from dirty_mkt_data.viz.themes import THEMES  # noqa: E402


def _slice_ohlcv(ohlcv, i: int, bars: int):
    return type(ohlcv)(
        opens=ohlcv.opens[i : i + 1, :bars],
        highs=ohlcv.highs[i : i + 1, :bars],
        lows=ohlcv.lows[i : i + 1, :bars],
        closes=ohlcv.closes[i : i + 1, :bars],
        vols=ohlcv.vols[i : i + 1, :bars],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="GBM vs AR(1) candlestick A/B PNG")
    ap.add_argument("--bars", type=int, default=100)
    ap.add_argument("--phi", type=float, default=0.7, help="AR(1) coefficient for the AR panel")
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--theme", default="synthwave", choices=tuple(THEMES))
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--invert", action="store_true", help="swap bullish/bearish bar colors")
    args = ap.parse_args()

    params = {args.symbol: (0.30, 0.60, 60_000.0)}
    gbm = generate(symbols=params, n_steps=args.bars, seed=args.seed,
                   low_tf=5, high_tf=5, regime="neutral", ar_coef=0.0)
    ar = generate(symbols=params, n_steps=args.bars, seed=args.seed,
                  low_tf=5, high_tf=5, regime="neutral", ar_coef=args.phi)

    theme = THEMES[args.theme]
    i = gbm.symbols.index(args.symbol)
    o_gbm = _slice_ohlcv(gbm.ohlcv, i, args.bars)
    o_ar = _slice_ohlcv(ar.ohlcv, i, args.bars)
    panels = [
        (f"{args.symbol} pure GBM (phi=0)", o_gbm, args.out / f"{args.symbol}_gbm_phi0.png"),
        (f"{args.symbol} AR(1) phi={args.phi:.2f}", o_ar,
         args.out / f"{args.symbol}_ar_phi{args.phi:+.2f}.png"),
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    for title, ohlcv, out_png in panels:
        fig = candle_figure(ohlcv, theme, path=0, title=title, show_volume=False)
        if args.invert:
            c = fig.data[0]
            up = (c.increasing.line.color, c.increasing.fillcolor)
            down = (c.decreasing.line.color, c.decreasing.fillcolor)
            c.increasing.line.color, c.increasing.fillcolor = down
            c.decreasing.line.color, c.decreasing.fillcolor = up
        fig.update_layout(
            template="plotly_dark", height=520,
            title=f"{title} — {args.bars} bars (seed {args.seed})",
        )
        fig.write_image(str(out_png), scale=args.scale)
        print(f"saved {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())