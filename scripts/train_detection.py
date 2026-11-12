#!/usr/bin/env python3
"""Train incident triage and monitoring outlier detection from the supplied CSVs."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.detection_experiment import run_detection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/detection")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/detection")
    parser.add_argument("--review-fraction", type=float, default=0.01, help="Calibration-tail review budget; not incident prevalence.")
    args = parser.parse_args()
    try:
        run_detection(args.data_dir, args.model_dir, args.report_dir, args.review_fraction)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Detection training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
