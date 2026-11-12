#!/usr/bin/env python3
"""Build a local FAISS index from runbooks and earlier incident records."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.knowledge_retrieval import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "models/knowledge")
    parser.add_argument("--cutoff")
    args = parser.parse_args()
    try:
        result = build_index(args.data_dir, args.index_dir, cutoff=args.cutoff)
        print(f"KNOWLEDGE INDEX: {result['rows']} passages; {result['dimensions']} TF-IDF dimensions")
        print(json.dumps(result["sources"], indent=2, allow_nan=False))
        print(f"Index: {args.index_dir}")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Knowledge indexing failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
