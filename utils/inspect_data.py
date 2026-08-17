"""Render synthetic crypto OHLCV candles to PNG (plotly + kaleido)."""

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


def main() -> int:
    ap = argparse.ArgumentParser(description="Candlestick PNGs of the synthetic universe")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--theme", default="synthwave", choices=tuple(THEMES))
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    b = generate(n_steps=args.steps, seed=args.seed)
    theme = THEMES[args.theme]

    for i, sym in enumerate(b.symbols):
        fig = candle_figure(b.ohlcv, theme, path=i, title=f"{sym}")
        fig.write_image(args.out / f"{sym}_{args.theme}.png", scale=args.scale)

    print(f"saved {len(b.symbols)} PNGs to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
