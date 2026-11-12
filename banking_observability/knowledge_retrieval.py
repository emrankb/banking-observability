"""Local lexical retrieval with source spans and an exact FAISS index."""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import platform
import tempfile

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .forecasting import file_sha256
from .knowledge_sources import KINDS, collect_knowledge, naive_time, source_report, validate_chunk

FAISS_VERSION = "1.13.2"
ENCODER_SETTINGS = {"lowercase": True, "strip_accents": "unicode", "stop_words": "english",
                    "ngram_range": (1, 2), "sublinear_tf": True, "norm": "l2"}
FILES = ("passages.jsonl", "encoder.joblib", "passages.faiss")


def require_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required. Install with .venv/bin/python -m pip install -r requirements-retrieval.txt") from exc
    if metadata.version("faiss-cpu") != FAISS_VERSION:
        raise ValueError(f"Expected faiss-cpu=={FAISS_VERSION}; install requirements-retrieval.txt.")
    return faiss


def runtime_versions() -> dict:
    return {"python": platform.python_version(), **{name: metadata.version(name) for name in
            ("numpy", "scipy", "scikit-learn", "joblib", "faiss-cpu")}}


def code_hashes() -> dict:
    return {name: file_sha256(Path(__file__).with_name(name)) for name in
            ("knowledge_sources.py", "knowledge_retrieval.py")}


def document_text(chunk: dict) -> str:
    # A short title supplies context to a section without repeating an entire document.
    return chunk["title"] + "\n" + chunk["text"]


def fit_encoder(chunks: list[dict]) -> tuple[TfidfVectorizer, np.ndarray]:
    if not chunks:
        raise ValueError("Cannot encode an empty corpus.")
    encoder = TfidfVectorizer(**ENCODER_SETTINGS, dtype=np.float32)
    matrix = np.ascontiguousarray(encoder.fit_transform([document_text(c) for c in chunks]).toarray(), dtype=np.float32)
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6):
        raise ValueError("Passage vectors must be finite, nonzero, and normalized.")
    return encoder, matrix


def build_index(data_dir: Path, index_dir: Path, *, cutoff: str | None = None) -> dict:
    """Create a new local index. Existing artifacts and source files are never replaced."""
    data_dir, index_dir = Path(data_dir).resolve(), Path(index_dir).resolve()
    if index_dir.exists() or index_dir == data_dir or index_dir in data_dir.parents or data_dir in index_dir.parents:
        raise ValueError("Use a new index directory separate from source data.")
    faiss = require_faiss()
    chunks, summary = collect_knowledge(data_dir, cutoff=cutoff)
    for chunk in chunks:
        validate_chunk(chunk)
    encoder, matrix = fit_encoder(chunks)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".knowledge-build-", dir=index_dir.parent) as scratch:
        staging = Path(scratch) / "index"
        staging.mkdir()
        (staging / "passages.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n" for c in chunks), encoding="utf-8")
        joblib.dump(encoder, staging / "encoder.joblib", compress=3)
        faiss.write_index(index, str(staging / "passages.faiss"))
        manifest = {"format_version": 1, "index_type": "IndexFlatIP", "embedding": "word_tfidf_unigrams_bigrams",
                    "neural_embeddings": False, "metric": "cosine_similarity", "dimensions": matrix.shape[1],
                    "rows": len(chunks), "sources": summary, "runtime_versions": runtime_versions(),
                    "code_sha256": code_hashes(), "artifact_sha256": {name: file_sha256(staging / name) for name in FILES}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        (staging / "source_review.md").write_text(source_report(summary))
        # Exercise serialization before publishing the directory.
        loaded = KnowledgeStore(staging)
        if loaded.index.ntotal != len(chunks):
            raise ValueError("Saved index row count changed.")
        if index_dir.exists():
            raise ValueError("Index destination appeared during build; refusing to overwrite it.")
        staging.rename(index_dir)
    return manifest


class KnowledgeStore:
    """Open trusted local index files; hashes detect changes, not a malicious publisher."""
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        faiss = require_faiss()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        m = self.manifest
        if m.get("format_version") != 1 or m.get("index_type") != "IndexFlatIP" or m.get("metric") != "cosine_similarity":
            raise ValueError("Unsupported knowledge-index format.")
        if m.get("runtime_versions") != runtime_versions() or m.get("code_sha256") != code_hashes():
            raise ValueError("Knowledge index differs from this code/runtime; rebuild it in this environment.")
        if set(m.get("artifact_sha256", {})) != set(FILES):
            raise ValueError("Incomplete knowledge-index file manifest.")
        for name in FILES:
            if file_sha256(self.directory / name) != m["artifact_sha256"][name]:
                raise ValueError(f"Knowledge artifact checksum mismatch: {name}")
        self.chunks = [json.loads(line) for line in (self.directory / "passages.jsonl").read_text().splitlines()]
        for chunk in self.chunks:
            validate_chunk(chunk)
        if len({c["chunk_id"] for c in self.chunks}) != len(self.chunks):
            raise ValueError("Duplicate chunk identifiers in the index.")
        self.cutoff = naive_time(m["sources"]["cutoff"])
        if any(c["event_timestamp"] and naive_time(c["event_timestamp"]) > self.cutoff for c in self.chunks):
            raise ValueError("Index contains an incident beyond its cutoff.")
        # Native indexes and joblib payloads are opened only after their checksums pass.
        self.encoder = joblib.load(self.directory / "encoder.joblib")
        self.index = faiss.read_index(str(self.directory / "passages.faiss"))
        if not isinstance(self.encoder, TfidfVectorizer):
            raise ValueError("Unexpected passage encoder type.")
        if not isinstance(self.index, faiss.IndexFlatIP) or self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Expected an exact inner-product index.")
        if self.index.ntotal != len(self.chunks) or self.index.ntotal != m["rows"] or self.index.d != m["dimensions"]:
            raise ValueError("Index shape does not match the document manifest.")
        if len(self.encoder.vocabulary_) != self.index.d:
            raise ValueError("Encoder dimension does not match the index.")

    def source_changes(self, data_dir: Path, *, notes: Path | None = None, catalog: Path | None = None) -> list[str]:
        from .knowledge_sources import ROOT
        notes = ROOT / "knowledge/runbooks.md" if notes is None else Path(notes)
        catalog = ROOT / "config/log_patterns.json" if catalog is None else Path(catalog)
        paths = {"knowledge/runbooks.md": notes, "config/log_patterns.json": catalog}
        return [logical for logical, expected in self.manifest["sources"]["source_sha256"].items()
                if not (path := paths.get(logical, Path(data_dir) / Path(logical).name)).is_file()
                or file_sha256(path) != expected]

    def get_document(self, document_id: str) -> list[dict]:
        rows = [chunk for chunk in self.chunks if chunk["document_id"] == document_id]
        if not rows:
            raise KeyError(f"Unknown document ID: {document_id}")
        return json.loads(json.dumps(rows))

    def search(self, query: str, *, top_k: int = 3, source_kind: str = "runbooks",
               min_similarity: float = 0.15, as_of: str | None = None) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ValueError("Supply a nonempty query of at most 4096 characters.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 through 20.")
        if not np.isfinite(min_similarity) or not 0 < min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 (exclusive) and 1 (inclusive).")
        if source_kind not in (*KINDS, "runbooks", "all"):
            raise ValueError("Unknown source-kind filter.")
        origin = self.cutoff if as_of is None else naive_time(as_of)
        if origin < self.cutoff:
            raise ValueError("as_of precedes the index cutoff; rebuild an earlier corpus for that replay.")
        allowed = set(KINDS) if source_kind == "all" else (
            {"supplied_runbook", "project_runbook"} if source_kind == "runbooks" else {source_kind})
        vector = np.ascontiguousarray(self.encoder.transform([query.strip()]).toarray(), dtype=np.float32)
        result = {"query": query.strip(), "method": "tfidf_faiss_cosine", "knowledge_cutoff": self.cutoff.isoformat(),
                  "as_of": origin.isoformat(), "source_kind": source_kind, "top_k": top_k,
                  "min_similarity": float(min_similarity), "status": "no_matches", "results": [],
                  "limitation": "Lexical relevance only; retrieved text is not a confirmed cause or an action authorization."}
        if np.linalg.norm(vector) == 0:
            result["status"] = "no_vocabulary_overlap"
            return result
        # The corpus is small. Search all rows so filtering cannot hide eligible top-k hits.
        scores, indices = self.index.search(vector, self.index.ntotal)
        candidates = [(float(score), self.chunks[int(i)]) for score, i in zip(scores[0], indices[0])
                      if i >= 0 and float(score) >= min_similarity and self.chunks[int(i)]["kind"] in allowed]
        candidates.sort(key=lambda pair: (-round(pair[0], 7), pair[1]["chunk_id"]))
        seen = set()
        for score, chunk in candidates:
            if chunk["document_id"] in seen:
                continue
            seen.add(chunk["document_id"])
            result["results"].append({"score": score, **json.loads(json.dumps(chunk))})
            if len(result["results"]) == top_k:
                break
        if result["results"]:
            result["status"] = "matches"
        return result
