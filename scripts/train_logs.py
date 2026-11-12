#!/usr/bin/env python3
"""Compare log clusters on earlier records and inspect the remaining messages."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.log_experiment import run_log_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/log_clustering")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/logs")
    parser.add_argument("--catalog", type=Path, default=ROOT / "config/log_patterns.json")
    args = parser.parse_args()
    try:
        run_log_experiment(args.data_dir, args.model_dir, args.report_dir, args.catalog)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Log training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
