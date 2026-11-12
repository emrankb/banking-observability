#!/usr/bin/env python3
"""Inspect a saved cluster assignment and an exact-template routing label."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.log_clustering import LogClassifier, read_logs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/log_clustering")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv-row", type=int, help="Original CSV line number; the header is row 1.")
    source.add_argument("--message", help="A new message to inspect, not training data.")
    parser.add_argument("--log-file", type=Path, default=ROOT / "data/raw/application_logs.csv")
    parser.add_argument("--log-level", help="Required with --message; kept separate from message wording.")
    args = parser.parse_args()
    try:
        if args.csv_row is not None:
            if args.log_level is not None:
                raise ValueError("Use the source log level with --csv-row; do not override it.")
            frame = read_logs(args.log_file)
            rows = frame.loc[frame.csv_row.eq(args.csv_row)]
            if rows.empty:
                raise ValueError(f"csv-row must be between 2 and {len(frame) + 1}.")
            row = rows.iloc[0]; message, level = row.log_message, row.log_level
            source_info = {"csv_row": int(row.csv_row), "timestamp": row.timestamp.isoformat(), "log_message": message, "log_level": level}
        else:
            if args.log_level is None:
                raise ValueError("--message requires --log-level.")
            message, level = args.message, args.log_level
            source_info = {"log_message": message, "log_level": level}
        result = LogClassifier(args.model_dir).predict(message, level)
        print(json.dumps({**source_info, **result}, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Log classification failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
