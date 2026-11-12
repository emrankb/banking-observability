#!/usr/bin/env python3
"""Score a monitoring minute using the preceding hour and the saved detector."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.anomaly_detection import MonitoringDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/detection/monitoring_anomaly")
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.add_argument("--as-of", help="Exact recorded timestamp; later observations are ignored.")
    args = parser.parse_args()
    try:
        history = pd.concat([pd.read_csv(path) for path in args.history], ignore_index=True)
        response = MonitoringDetector(args.model_dir).predict(history, as_of=args.as_of)
        print(json.dumps(response, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Monitoring scoring failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
