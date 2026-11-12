#!/usr/bin/env python3
"""Predict a five-minute-ahead resource or payment success-rate reading."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.kpi_forecasting import KPIForecaster, TARGETS
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, choices=TARGETS)
    parser.add_argument('--history', type=Path, default=ROOT / 'data/raw/infrastructure_metrics.csv')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/kpi_5min')
    parser.add_argument('--as-of')
    args = parser.parse_args()
    try:
        history = pd.read_csv(args.history)
        forecaster = KPIForecaster(args.model_dir / args.target)
        result = forecaster.predict(history, as_of=args.as_of)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'KPI prediction failed: {exc}\n')


if __name__ == '__main__':
    main()
