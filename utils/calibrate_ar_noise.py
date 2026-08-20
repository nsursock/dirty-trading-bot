"""Grid-search ``data.ar_noise`` so phi=0.7 lands in a target Sharpe band.

    caffeinate -dims venv/bin/python utils/calibrate_ar_noise.py \\
        --target-lo 2 --target-hi 5 --kappas 1.65,1.67,1.69,1.71

``ar_noise`` is a constant kappa across phi (calibrate at phi=0.7).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def _run(kappa: float, timesteps: int, config: Path) -> dict:
    cfg = yaml.safe_load(config.read_text())
    cfg.setdefault("data", {})["ar"] = 0.7
    cfg["data"]["ar_noise"] = float(kappa)
    cfg.setdefault("eval", {})["deterministic"] = True
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=ROOT / "logs") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
        cfg_path = Path(fh.name)
    log_path = cfg_path.with_suffix(".log")
    try:
        with open(log_path, "w") as log_fh:
            subprocess.run(
                [sys.executable, "scripts/main.py", "full", "--config", str(cfg_path),
                 "--timesteps", str(timesteps)],
                cwd=ROOT, stdout=log_fh, stderr=subprocess.STDOUT, check=True,
            )
        for line in log_path.read_text().splitlines():
            if line.startswith("done ->"):
                run_testing = Path(line.split("->")[1].split()[0].strip())
                rep = json.loads((run_testing.parent / "report.json").read_text())
                pm = rep["portfolio"]
                return {
                    "kappa": kappa,
                    "sharpe": pm.get("sharpe"),
                    "total_return": pm.get("total_return"),
                    "verdict": rep["plausibility"]["status"],
                }
        raise RuntimeError(f"no done line in {log_path}")
    finally:
        cfg_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "exp_ar.yaml"))
    ap.add_argument("--timesteps", type=int, default=750_000)
    ap.add_argument("--target-lo", type=float, default=2.0)
    ap.add_argument("--target-hi", type=float, default=5.0)
    ap.add_argument("--kappas", default="1.65,1.67,1.69,1.71,1.73",
                    help="comma-separated kappa values to try")
    args = ap.parse_args()
    kappas = [float(x) for x in args.kappas.split(",") if x.strip()]
    config = Path(args.config)

    print(f"calibrate: phi=0.7  target Sharpe [{args.target_lo}, {args.target_hi}]")
    rows = []
    for k in tqdm(kappas, desc="kappa grid", unit="run"):
        row = _run(k, args.timesteps, config)
        rows.append(row)
        sh = row["sharpe"]
        shs = "nan" if sh is None else f"{sh:+.2f}"
        in_band = sh is not None and args.target_lo <= sh <= args.target_hi
        flag = " <--" if in_band else ""
        tqdm.write(f"kappa={k:.2f}  sharpe={shs}  ret={row['total_return']:+.4f}{flag}")

    hits = [r for r in rows if r["sharpe"] is not None
            and args.target_lo <= r["sharpe"] <= args.target_hi]
    if hits:
        best = min(hits, key=lambda r: abs(r["sharpe"] - (args.target_lo + args.target_hi) / 2))
        print(f"\nbest in band: ar_noise={best['kappa']:.2f}  sharpe={best['sharpe']:+.2f}")
    else:
        print("\nno kappa in target band — widen grid or adjust timesteps")


if __name__ == "__main__":
    main()
