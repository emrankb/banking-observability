"""Group log wording without treating a text cluster as an incident label."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits

from .detection_common import load_detector

NORMALIZER_VERSION = 1
TIME = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?", re.I)
PREFIX = re.compile(r"^\s*(?:\[?<TIME>\]?\s*)?(INFO|WARNING|WARN|ERROR|CRITICAL|DEBUG|FATAL)\s*:\s*", re.I)
NUMBER = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?")
STOP_WORDS = ["id", "on", "in", "for", "the", "to", "current", "reason", "server", "time", "ms"]
VECTOR_PARAMS = {"ngram_range": (1, 2), "norm": "l2", "sublinear_tf": True,
                 "token_pattern": r"(?u)\b[a-z][a-z_/]+\b", "stop_words": STOP_WORDS}
SEED_OFFSETS = (0, 11, 29)
KMEANS_PARAMS = {"n_init": 20, "max_iter": 500, "algorithm": "lloyd", "init": "k-means++", "tol": 1e-4}
REQUIRED = ("timestamp", "log_level", "log_message")
CATEGORIES = {"infrastructure", "business_kpi", "routine"}


def normalize_message(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("log_message must be a nonempty string.")
    text = TIME.sub("<TIME>", message.strip())
    text = PREFIX.sub("", text)
    for pattern, replacement in ((r"\bTXN\d+\b", "<TXN_ID>"), (r"\bUSR\d+\b", "<USER_ID>"),
                                  (r"\bsrv-\d+\b", "<HOST_ID>")):
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = NUMBER.sub("<NUM>", text)
    text = " ".join(text.split()).lower()
    if not re.search(r"[a-z]", lexical_text(text)):
        raise ValueError("No message words remain after normalization.")
    return text


def lexical_text(text: str) -> str:
    # Placeholders describe structure; they must not join unrelated events by ID type.
    return re.sub(r"<[^>]+>", " ", text)


def read_logs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or frame.columns.duplicated().any() or not set(REQUIRED).issubset(frame):
        raise ValueError("Log CSV requires timestamp, log_level, and log_message columns and at least one row.")
    if frame.loc[:, REQUIRED].isna().any().any():
        raise ValueError("Log timestamps, levels, and messages must not be missing.")
    result = frame.loc[:, REQUIRED].copy()
    time = pd.to_datetime(result.timestamp, errors="raise", format="ISO8601")
    if not pd.api.types.is_datetime64_any_dtype(time) or time.dt.tz is not None or not time.is_monotonic_increasing:
        raise ValueError("Logs must be in nondecreasing order on the timezone-naive source clock.")
    for col in ("log_level", "log_message"):
        if not result[col].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{col} must contain nonempty strings.")
    result["timestamp"] = time
    result.insert(0, "csv_row", np.arange(2, len(result) + 2))
    result["normalized_message"] = result.log_message.map(normalize_message)
    result["duplicate_after_first"] = result.duplicated(list(REQUIRED), keep="first")
    return result


def read_catalog(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("format_version") != 1 or not isinstance(data.get("patterns"), list):
        raise ValueError("Unsupported log-pattern catalog.")
    catalog = {}
    for entry in data["patterns"]:
        if not isinstance(entry, dict) or set(entry) != {"template", "event_type", "category"}:
            raise ValueError("Each catalog entry needs template, event_type, and category.")
        if entry["category"] not in CATEGORIES or not isinstance(entry["event_type"], str) or not entry["event_type"].strip():
            raise ValueError("Invalid event type or catalog category.")
        key = normalize_message(entry["template"])
        if key in catalog:
            raise ValueError("Duplicate normalized template in catalog.")
        catalog[key] = {"category": entry["category"], "event_type": entry["event_type"]}
    if not catalog:
        raise ValueError("Catalog is empty.")
    return catalog


def message_evidence(message: str, log_level: str) -> dict:
    if not isinstance(log_level, str) or not log_level.strip():
        raise ValueError("log_level must be a nonempty string.")
    prefix = PREFIX.match(TIME.sub("<TIME>", message.strip()))
    level = prefix.group(1).upper() if prefix else None
    flags, readings = [], {}
    if level is not None and level != log_level.strip().upper():
        flags.append("structured_message_severity_mismatch")
    for metric, phrase in (("cpu_pct", "high cpu utilization"), ("memory_pct", "memory usage")):
        if phrase in message.lower():
            match = re.search(r"current:\s*([-+]?\d+(?:\.\d+)?)\s*%", message, re.I)
            if match:
                value = float(match.group(1)); readings[metric] = value
                if not 0 <= value <= 100:
                    flags.append(f"{metric}_outside_0_100_scope_unverified")
    return {"message_level": level, "quality_flags": flags, "logged_resource_readings": readings}


def template_seed(templates: list[str]) -> int:
    return int(hashlib.sha256("\n".join(sorted(templates)).encode()).hexdigest()[:8], 16)


@dataclass
class LogClusterModel:
    vectorizer: TfidfVectorizer
    estimator: KMeans
    templates: list[str]
    template_clusters: list[int]
    raw_to_cluster: dict[int, int]
    catalog: dict[str, dict]
    profiles: list[dict]
    fit_through: str
    fit_levels: list[str]

    def classify(self, message: str, log_level: str) -> dict:
        normalized = normalize_message(message)
        evidence = message_evidence(message, log_level)
        vector = self.vectorizer.transform([normalized])
        known = normalized in self.templates
        cluster = None
        nearest = None
        if vector.nnz:
            with threadpool_limits(limits=1):
                raw_cluster = int(self.estimator.predict(vector)[0])
            cluster = self.raw_to_cluster[raw_cluster]
            scores = cosine_similarity(vector, self.vectorizer.transform(self.templates))[0]
            closest = int(np.argmax(scores))
            nearest = {"normalized_message": self.templates[closest], "cosine_similarity": float(scores[closest])}
        mapping = self.catalog.get(normalized) if known else None
        # No guessed event label for wording that was absent from fitting.
        needs_review = not known or mapping is None or bool(evidence["quality_flags"])
        if log_level.strip().upper() not in self.fit_levels:
            evidence["quality_flags"].append("structured_level_not_seen_during_fit")
            needs_review = True
        profile = self.profiles[cluster] if cluster is not None else None
        return {"normalized_message": normalized, "template_seen_during_fit": known,
                "cluster_id": cluster, "cluster_label": profile["label"] if profile else None,
                "cluster_categories": profile["categories"] if profile else [],
                "cluster_mixes_categories": profile["mixed_categories"] if profile else None,
                "category": mapping["category"] if mapping else "unreviewed",
                "event_type": mapping["event_type"] if mapping else None,
                "category_source": "exact_normalized_template_catalog" if mapping else "no_exact_fitted_template",
                "nearest_training_template": nearest, "label_review_required": needs_review, **evidence,
                "scope": "log_message_grouping", "trained_through": self.fit_through}


def fit_log_clusters(development: pd.DataFrame, catalog: dict[str, dict]):
    templates = sorted(development.normalized_message.unique())
    if not 3 <= len(templates) <= 100:
        raise ValueError("This template-level experiment requires 3 to 100 distinct training patterns.")
    vectorizer = TfidfVectorizer(preprocessor=lexical_text, **VECTOR_PARAMS)
    X = vectorizer.fit_transform(templates)
    if np.any(X.getnnz(axis=1) == 0):
        raise ValueError("A training pattern has no usable vocabulary.")
    seed = template_seed(templates)
    rows, details = [], []
    with threadpool_limits(limits=1):
        for k in range(2, len(templates)):
            labels, scores = [], []
            for offset in SEED_OFFSETS:
                model = KMeans(n_clusters=k, random_state=(seed + offset) % (2**32), **KMEANS_PARAMS).fit(X)
                if len(set(model.labels_)) != k:
                    raise ValueError("Distinct normalized patterns produced fewer clusters than requested.")
                value = float(silhouette_score(X, model.labels_, metric="cosine"))
                scores.append(value); labels.append(model.labels_)
                details.append({"k": k, "seed_offset": offset, "silhouette": value,
                                "inertia": float(model.inertia_), "iterations": int(model.n_iter_),
                                "singleton_template_clusters": int((np.bincount(model.labels_) == 1).sum())})
            ari = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
            rows.append({"k": k, "mean_template_silhouette": float(np.mean(scores)),
                         "min_template_silhouette": float(min(scores)), "mean_seed_ari": float(np.mean(ari))})
        comparison = pd.DataFrame(rows).sort_values(
            ["mean_template_silhouette", "mean_seed_ari", "k"], ascending=[False, False, True]).reset_index(drop=True)
        selected_k = int(comparison.iloc[0].k)
        estimator = KMeans(n_clusters=selected_k, random_state=seed, **KMEANS_PARAMS).fit(X)
    # Stable display numbering, rather than assigning meaning to arbitrary KMeans IDs.
    groups = {i: [t for t, label in zip(templates, estimator.labels_) if label == i] for i in range(selected_k)}
    order = sorted(groups, key=lambda i: min(groups[i]))
    raw_to_cluster = {raw: new for new, raw in enumerate(order)}
    template_clusters = [raw_to_cluster[int(i)] for i in estimator.labels_]
    fitted_catalog = {key: value for key, value in catalog.items() if key in templates}
    vocabulary = vectorizer.get_feature_names_out()
    profiles = []
    for raw in order:
        members = groups[raw]
        weights = estimator.cluster_centers_[raw]
        ranked = sorted(range(len(weights)), key=lambda j: (-weights[j], vocabulary[j]))
        terms = [str(vocabulary[j]) for j in ranked if weights[j] > 0][:6]
        categories = sorted({fitted_catalog.get(t, {}).get("category", "unreviewed") for t in members})
        profiles.append({"cluster_id": raw_to_cluster[raw], "label": " / ".join(terms[:3]),
                         "top_terms": terms, "templates": members, "categories": categories,
                         "mixed_categories": len(categories) > 1, "template_count": len(members)})
    fitted = LogClusterModel(vectorizer, estimator, templates, template_clusters, raw_to_cluster,
                             fitted_catalog, profiles, str(development.timestamp.max()),
                             sorted(development.log_level.str.upper().unique()))
    plan = {"selected_k": selected_k, "training_templates": len(templates), "feature_count": X.shape[1],
            "candidate_k": list(range(2, len(templates))), "seed": seed, "seed_offsets": list(SEED_OFFSETS),
            "kmeans_parameters": KMEANS_PARAMS, "vectorizer_parameters": {**VECTOR_PARAMS, "ngram_range": list(VECTOR_PARAMS["ngram_range"])},
            "normalizer_version": NORMALIZER_VERSION,
            "selection_rule": "Mean cosine silhouette on distinct training templates, then mean seed ARI, then smaller k.",
            "final_template_silhouette": float(silhouette_score(X, estimator.labels_, metric="cosine")),
            "training_end": fitted.fit_through, "training_rows": len(development),
            "scope": "Lexical groups, not learned incident classes; category tags use an explicit template catalog."}
    return fitted, plan, {"cluster_selection": comparison, "cluster_seed_results": pd.DataFrame(details)}


class LogClassifier:
    def __init__(self, directory: Path):
        self.model, self.metadata = load_detector(directory, kind="log_clustering", expected_type=LogClusterModel)
        if self.metadata.get("normalizer_version") != NORMALIZER_VERSION:
            raise ValueError("Normalizer version differs from the saved artifact.")

    def predict(self, message: str, log_level: str) -> dict:
        return {**self.model.classify(message, log_level), "model_sha256": self.metadata["model_sha256"]}
