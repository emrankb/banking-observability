#!/usr/bin/env python3
"""Check source retrieval against a small, explicitly authored query set."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.knowledge_retrieval import KnowledgeStore
from banking_observability.forecasting import file_sha256


def evaluate(store: KnowledgeStore, questions: dict) -> dict:
    items = questions.get("queries")
    if not isinstance(items, list) or not items or not any(item.get("expected_document") for item in items):
        raise ValueError("Provide a nonempty query list with at least one expected document.")
    known_documents = {chunk["document_id"] for chunk in store.chunks}
    rows = []
    for item in items:
        expected = item.get("expected_document")
        if expected is not None and expected not in known_documents:
            raise ValueError(f"Expected document is not indexed: {expected}")
        reply = store.search(item["query"], top_k=questions["top_k"], min_similarity=questions["min_similarity"])
        retrieved = [hit["document_id"] for hit in reply["results"]]
        expected = item["expected_document"]
        rank = retrieved.index(expected) + 1 if expected in retrieved else None
        rows.append({"query": item["query"], "expected_document": expected, "retrieved_documents": retrieved,
                     "rank": rank, "passed": not retrieved if expected is None else rank is not None,
                     "reciprocal_rank": 1.0 / rank if rank else 0.0, "status": reply["status"]})
    positives = [r for r in rows if r["expected_document"] is not None]
    negatives = [r for r in rows if r["expected_document"] is None]
    return {"scope": questions["description"], "top_k": questions["top_k"], "min_similarity": questions["min_similarity"],
            "answerable_queries": len(positives), "hit_rate_at_k": sum(r["passed"] for r in positives)/len(positives),
            "mean_reciprocal_rank_at_k": sum(r["reciprocal_rank"] for r in positives)/len(positives),
            "no_match_queries": len(negatives), "correct_no_match_count": sum(r["passed"] for r in negatives), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "models/knowledge")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--queries", type=Path, default=ROOT / "config/retrieval_queries.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/retrieval")
    args = parser.parse_args()
    try:
        destination, data, index = (p.resolve() for p in (args.output_dir, args.data_dir, args.index_dir))
        if destination.exists() or any(destination == p or destination in p.parents or p in destination.parents for p in (data,index)):
            raise ValueError("Use a new report directory separate from data and index files.")
        store = KnowledgeStore(index)
        changes = store.source_changes(data)
        if changes:
            raise ValueError("Knowledge sources changed: " + ", ".join(changes))
        questions = json.loads(args.queries.read_text())
        result = {**evaluate(store, questions), "query_file_sha256": file_sha256(args.queries),
                  "index_manifest_sha256": file_sha256(index / "manifest.json")}
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "query_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        lines = ["# Retrieval checks", "", result["scope"], "",
                 f"Hit rate at {result['top_k']}: {result['hit_rate_at_k']:.4f} on {result['answerable_queries']} authored queries.",
                 f"Mean reciprocal rank at {result['top_k']}: {result['mean_reciprocal_rank_at_k']:.4f}.",
                 f"Correct no-match checks: {result['correct_no_match_count']} of {result['no_match_queries']}.", "",
                 "| Query | Expected document | Rank |", "|---|---|---:|"]
        lines.extend(f"| {r['query']} | {r['expected_document'] or 'no result'} | {r['rank'] or 'none'} |" for r in result['rows'])
        lines += ["", "These checks cover known topics with shared wording. They do not measure general semantic search or root-cause accuracy.", ""]
        (destination / "retrieval_review.md").write_text("\n".join(lines))
        print(json.dumps(result, indent=2, allow_nan=False))
        print(f"Report: {destination / 'retrieval_review.md'}")
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Retrieval evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
