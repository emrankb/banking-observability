#!/usr/bin/env python3
"""Inspect knowledge passages and source references before vector indexing."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.knowledge_sources import preview_sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/knowledge_sources")
    parser.add_argument("--cutoff")
    args = parser.parse_args()
    try:
        result = preview_sources(args.data_dir, args.output_dir, cutoff=args.cutoff)
        print(json.dumps(result, indent=2, allow_nan=False))
        print(f"Report: {args.output_dir / 'source_review.md'}")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Knowledge inspection failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
