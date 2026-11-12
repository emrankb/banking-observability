#!/usr/bin/env python3
"""Retrieve source passages without generating an incident explanation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.knowledge_retrieval import KnowledgeStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "models/knowledge")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--source-kind", default="runbooks")
    parser.add_argument("--min-similarity", type=float, default=0.15)
    parser.add_argument("--as-of")
    args = parser.parse_args()
    try:
        result = KnowledgeStore(args.index_dir).search(args.query, top_k=args.top_k,
                    source_kind=args.source_kind, min_similarity=args.min_similarity, as_of=args.as_of)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Knowledge search failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
