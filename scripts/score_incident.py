#!/usr/bin/env python3
"""Score one supplied incident record with the saved triage model."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.incident_triage import IncidentTriage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/detection/incident_triage")
    parser.add_argument("--incidents", type=Path, default=ROOT / "data/raw/incident_reports.csv")
    parser.add_argument("--incident-id", required=True)
    args = parser.parse_args()
    try:
        records = pd.read_csv(args.incidents)
        row = records.loc[records.incident_id.eq(args.incident_id)]
        if len(row) != 1:
            raise ValueError("incident-id must identify exactly one source record.")
        response = IncidentTriage(args.model_dir).predict(row.iloc[0].to_dict())
        print(json.dumps(response, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
        parser.exit(2, f"Incident scoring failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
