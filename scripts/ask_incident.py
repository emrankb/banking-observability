#!/usr/bin/env python3
"""Create a local incident briefing, optionally with a cited model draft."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.incident_assistant import run_assistant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", help="Remote model ID; omitted for local evidence-only mode.")
    parser.add_argument("--allow-external-data", action="store_true",
                        help="Allow bounded monitoring, normalized logs, and runbook text to be sent to the API.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--knowledge-dir", type=Path, default=ROOT / "models/knowledge")
    parser.add_argument("--log-model-dir", type=Path, default=ROOT / "models/log_clustering")
    parser.add_argument("--anomaly-model-dir", type=Path, default=ROOT / "models/detection/monitoring_anomaly")
    parser.add_argument("--include-historical-incidents", action="store_true")
    args = parser.parse_args()
    try:
        result = run_assistant(question=args.question, as_of=args.as_of,
            data_dir=args.data_dir, knowledge_dir=args.knowledge_dir,
            log_model_dir=args.log_model_dir, anomaly_model_dir=args.anomaly_model_dir,
            model=args.model, allow_external_data=args.allow_external_data,
            include_historical=args.include_historical_incidents)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, AttributeError) as exc:
        parser.exit(2, f"Incident assistant failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
