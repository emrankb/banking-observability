#!/usr/bin/env python3
"""Fit five-minute resource and payment success-rate forecasts."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.kpi_experiment import train_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/kpi_5min')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'reports/kpi_forecast')
    args = parser.parse_args()
    try:
        train_all(args.data_dir, args.model_dir, args.report_dir)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f'KPI forecast training failed: {exc}\n')


if __name__ == '__main__':
    main()
