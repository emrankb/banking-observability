#!/usr/bin/env python3
"""Run the five-minute aggregate-processing-time experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.forecast_experiment import train_and_evaluate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/forecast")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/latency_5min")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing checkpoint.")
    args = parser.parse_args()
    try:
        train_and_evaluate(args.data_dir, args.output_dir, args.models_dir, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Forecast training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
