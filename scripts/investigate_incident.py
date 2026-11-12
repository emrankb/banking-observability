#!/usr/bin/env python3
"""Inspect a historical monitoring minute with logs and retrieved runbooks."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.incident_evidence import assemble_investigation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="Observed monitoring minute, timezone-naive.")
    parser.add_argument("--question", required=True, help="Investigation question used for retrieval.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--knowledge-dir", type=Path, default=ROOT / "models/knowledge")
    parser.add_argument("--log-model-dir", type=Path, default=ROOT / "models/log_clustering")
    parser.add_argument("--anomaly-model-dir", type=Path, default=ROOT / "models/detection/monitoring_anomaly")
    parser.add_argument("--lookback-minutes", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-historical-incidents", action="store_true")
    args = parser.parse_args()
    try:
        result = assemble_investigation(
            args.data_dir, question=args.question, as_of=args.as_of,
            index_dir=args.knowledge_dir, log_model_dir=args.log_model_dir,
            anomaly_model_dir=args.anomaly_model_dir,
            lookback_minutes=args.lookback_minutes, max_results=args.top_k,
            include_incident_snapshots=args.include_historical_incidents)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, AttributeError) as exc:
        parser.exit(2, f"Investigation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
