# Knowledge retrieval

This component searches local runbook and incident passages and returns source
references. Incident analysis and operational actions are handled separately.

## Sources

`incident_knowledge_base.txt` supplies five runbooks. Their wording is preserved.
`knowledge/runbooks.md` adds project investigation notes for the 13 non-routine
event types in `config/log_patterns.json`. These notes are not completed incident
reports or verified fixes.

The corpus also contains incident snapshots through the last timestamp in
`infrastructure_metrics_train.csv`. Each card keeps the supplied labels and KPIs,
compares them with the monitoring row at exactly the same timestamp, and cites
both rows. Conflicts are carried forward rather than reconciled. The source has
no resolution or confirmed-cause field; the cards do not add one.

The cutoff is recorded in the index. A query with `--as-of` before that cutoff is
rejected because the vocabulary and IDF weights already use the indexed corpus.
Build an earlier corpus with `--cutoff` for an earlier replay. Runbooks have no
publication or effective dates, so their availability at a historical timestamp
is unknown. They are treated as reference material, not time-verified evidence.

## Passages and citations

Runbooks are split at section headings, then at line boundaries when a section
exceeds 1,400 characters. A long section overlaps one line across consecutive
passages. The limit keeps excerpts small while retaining whole source lines.
Incident cards stay together so their KPI discrepancies are not separated from
the values being compared.

Each passage has a document ID, a content-derived chunk ID, source paths, line
ranges, source hashes, and exact source excerpts. Runbook text is verbatim;
incident-card text is a formatted summary of cited CSV records. Moving the
workspace does not change logical source paths.

## Vector search

This version uses local TF-IDF vectors: word unigrams and bigrams, sublinear term
frequency, English stop-word removal, and L2 normalization. These are lexical
vectors, not pretrained neural embeddings. There is no embedding service or
model download. A query using different vocabulary can miss a relevant passage.

FAISS `IndexFlatIP` stores the dense float32 vectors. With unit-length vectors,
inner product gives cosine similarity. The corpus is small enough for exact
search; no approximate index or dimensionality reduction is used.

The default query searches only supplied and project runbooks. Incident cards
are available with `--source-kind incident_snapshot`; `all` includes both. The
index is searched completely before filtering so a filter does not remove all
of an already-truncated top-k list. Ties are ordered by chunk ID. Results contain
at most one passage per document; `get_document()` expands all of its sections.

Search defaults to three documents and a minimum cosine similarity of 0.15.
The threshold is a starting heuristic. Queries without shared vocabulary return
no results; other matches still need a relevance check.

## Index files

`models/knowledge/` contains the FAISS index, fitted encoder, passages, source
inventory, and manifest. Existing directories are not overwritten. The manifest
records versions and file hashes; loading rejects mismatches. Open trusted local
artifacts only: hashes do not make unknown FAISS or joblib files safe to load.
Search uses the saved source snapshot; evaluation checks the current source hashes.
When sources or code change, build a new index instead of silently reusing one.

## Checks

`test_knowledge_sources.py` validates parsing, source spans, incident comparisons,
time cutoffs, and lexical encoding without loading FAISS.
`test_retrieval.py` requires the real FAISS package. It checks serialized search,
filtering, no-match behavior, reproducibility, and command-line inference.
The suite exits with an error if FAISS is unavailable.

`config/retrieval_queries.json` contains authored questions about the known topics
and an out-of-vocabulary query. `evaluate_retrieval.py` reports hit rate and mean
reciprocal rank at three, with per-query results. These known-topic questions
are a development check, not an independent
retrieval benchmark. They do not measure root-cause accuracy or performance on
new question wording.

## References

- [FAISS index types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [FAISS index serialization](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning)
- [scikit-learn TF-IDF vectorizer](https://scikit-learn.org/1.8/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
