#!/usr/bin/env python3
"""Inspect the supplied banking CSV files without modifying source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

FILES = (
    'transactions_fraud_train.csv',
    'infrastructure_metrics.csv',
    'application_logs.csv',
)


def inspect(path: Path) -> dict:
    frame = pd.read_csv(path)
    result = {
        'file': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows': len(frame),
        'columns': len(frame.columns),
        'duplicate_rows': int(frame.duplicated().sum()),
        'schema': [],
    }
    for name in frame.columns:
        s = frame[name]
        detail = {
            'column': name,
            'csv_inferred_dtype': str(s.dtype),
            'missing': int(s.isna().sum()),
            'unique_non_null': int(s.nunique()),
        }
        if s.dtype == 'object':
            detail['blank_strings'] = int(s.fillna('').str.strip().eq('').sum() - s.isna().sum())
        if 'timestamp' in name.lower() or name.lower().endswith('_date'):
            parsed = pd.to_datetime(s, errors='coerce')
            detail.update({
                'parsed_dtype': str(parsed.dtype),
                'invalid_timestamps': int((s.notna() & parsed.isna()).sum()),
                'min': str(parsed.min()),
                'max': str(parsed.max()),
                'monotonic_increasing': bool(parsed.is_monotonic_increasing),
                'duplicate_timestamps': int(parsed.duplicated().sum()),
                'timezone': str(parsed.dt.tz),
            })
            intervals = parsed.sort_values().drop_duplicates().diff().dt.total_seconds().dropna()
            detail['interval_seconds_counts'] = {str(k): int(v) for k, v in intervals.value_counts().items()}
        elif pd.api.types.is_numeric_dtype(s):
            detail['numeric_summary'] = {str(k): float(v) for k, v in s.describe(percentiles=[.01, .05, .5, .95, .99]).items()}
            if s.nunique() <= 20:
                detail['value_counts'] = {str(k): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}
        elif s.nunique() <= 100:
            detail['value_counts'] = {str(k): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}
        else:
            detail['examples'] = s.dropna().head(5).tolist()
        result['schema'].append(detail)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    missing = [name for name in FILES if not (args.data_dir / name).is_file()]
    if missing:
        parser.error('Missing input files: ' + ', '.join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [inspect(args.data_dir / name) for name in FILES]
    output = json.dumps(profiles, indent=2, allow_nan=False)
    (args.output_dir / 'schema.json').write_text(output + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
