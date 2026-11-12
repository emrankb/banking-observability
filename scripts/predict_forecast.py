#!/usr/bin/env python3
"""Load the saved model and forecast from real, ordered monitoring history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.forecasting import LatencyForecaster  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/latency_5min")
    parser.add_argument("--history", type=Path, action="append", required=True, help="Repeat for consecutive CSV files, in time order.")
    parser.add_argument("--as-of", help="Exact recorded origin; later history is ignored. Defaults to final row.")
    parser.add_argument("--output", type=Path, help="Also write the JSON response to this file.")
    args = parser.parse_args()
    try:
        model = LatencyForecaster(args.model_dir)
        history = pd.concat([pd.read_csv(path) for path in args.history], ignore_index=True)
        response = model.predict(history, as_of=args.as_of)
        rendered = json.dumps(response, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered, end="")
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"Forecast inference failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
