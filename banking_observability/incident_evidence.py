"""Collect a time-bounded evidence packet for an operational investigation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .detection_common import MONITORING_METRICS, metric_frame
from .forecasting import file_sha256
from .knowledge_sources import naive_time
from .log_clustering import normalize_message

MONITORING_FILE = "infrastructure_metrics.csv"
LOG_FILE = "application_logs.csv"
BASELINE_MINUTES = 60
MAX_LOG_ROWS = 20
MAX_LOOKBACK_MINUTES = 120
# The display order is fixed; the source CSV and the detector have more fields.
DISPLAY_METRICS = (
    "avg_transaction_processing_time_ms", "application_error_rate",
    "neft_success_rate", "imps_success_rate", "rtgs_success_rate",
    "auth_success_rate", "cpu_utilization", "memory_usage",
    "db_connection_pool_usage", "service_uptime",
)


def read_evidence(data_dir: Path, *, as_of: str, lookback_minutes: int = 30) -> dict:
    """Read existing CSV records only; neither incident labels nor values are inferred."""
    if isinstance(lookback_minutes, bool) or not isinstance(lookback_minutes, int) or not 1 <= lookback_minutes <= MAX_LOOKBACK_MINUTES:
        raise ValueError(f"lookback_minutes must be from 1 through {MAX_LOOKBACK_MINUTES}.")
    origin = naive_time(as_of)
    if origin.second or origin.microsecond:
        raise ValueError("as_of must match a recorded minute boundary.")
    root = Path(data_dir).resolve()
    metrics_path, logs_path = root / MONITORING_FILE, root / LOG_FILE
    if not metrics_path.is_file() or not logs_path.is_file():
        raise FileNotFoundError("Expected data/raw/infrastructure_metrics.csv and data/raw/application_logs.csv.")

    raw_metrics = pd.read_csv(metrics_path)
    if "timestamp" not in raw_metrics:
        raise ValueError("Monitoring CSV is missing timestamp.")
    raw_times = pd.to_datetime(raw_metrics.timestamp, format="ISO8601", errors="raise")
    if raw_times.isna().any() or raw_times.dt.tz is not None:
        raise ValueError("Monitoring timestamps must use the source clock.")
    # Future rows are not inputs to the snapshot or its baseline.
    metrics = metric_frame(raw_metrics.loc[raw_times.le(origin)], MONITORING_METRICS, step_minutes=1)
    matched = metrics.index[metrics.timestamp.eq(origin)]
    if len(matched) != 1:
        raise ValueError("as_of must match exactly one recorded monitoring minute.")
    position = int(matched[0])
    if position < BASELINE_MINUTES:
        raise ValueError("At least 60 earlier monitoring readings are required.")
    history = metrics.iloc[position - BASELINE_MINUTES:position + 1]
    if history.timestamp.iloc[-1] - history.timestamp.iloc[0] != pd.Timedelta(minutes=BASELINE_MINUTES):
        raise ValueError("A continuous 61-minute monitoring window is required.")
    previous, current = history.iloc[:-1], history.iloc[-1]
    monitoring_reference = {"path": f"data/raw/{MONITORING_FILE}", "sha256": file_sha256(metrics_path),
                            "csv_row": position + 2, "timestamp": origin.isoformat()}
    metric_evidence = []
    for metric in DISPLAY_METRICS:
        observed = float(current[metric]); median = float(previous[metric].median())
        if not np.isfinite([observed, median]).all():
            raise ValueError("A monitoring value is not finite.")
        metric_evidence.append({"metric": metric, "observed": observed,
                                "prior_60_minute_median": median, "difference_from_median": observed - median,
                                "reference": monitoring_reference})

    logs = pd.read_csv(logs_path)
    required = {"timestamp", "log_level", "log_message"}
    if logs.empty or logs.columns.duplicated().any() or not required.issubset(logs.columns):
        raise ValueError("Log CSV requires timestamp, log_level, and log_message.")
    log_times = pd.to_datetime(logs.timestamp, format="ISO8601", errors="raise")
    if log_times.isna().any() or log_times.dt.tz is not None:
        raise ValueError("Log timestamps must use the source clock.")
    start = origin - pd.Timedelta(minutes=lookback_minutes)
    selected = logs.loc[log_times.between(start, origin, inclusive="both")].copy()
    if selected[["log_level", "log_message"]].isna().any().any() or not selected.index.is_monotonic_increasing:
        raise ValueError("Selected log records have missing fields or invalid order.")
    selected.insert(0, "csv_row", selected.index + 2)
    selected["timestamp"] = log_times.loc[selected.index]
    if not selected.timestamp.is_monotonic_increasing:
        raise ValueError("Log source timestamps must be in nondecreasing order.")
    for field in ("log_level", "log_message"):
        if not selected[field].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Selected logs require nonempty {field}.")
    selected["normalized_message"] = selected.log_message.map(normalize_message)
    # The most recent records are kept. All source rows stay in data/raw.
    earlier = selected
    chosen = earlier.tail(MAX_LOG_ROWS).iloc[::-1]
    log_reference = {"path": f"data/raw/{LOG_FILE}", "sha256": file_sha256(logs_path)}
    messages = [{"csv_row": int(row.csv_row), "timestamp": row.timestamp.isoformat(),
                 "log_level": row.log_level, "log_message": row.log_message,
                 "normalized_message": row.normalized_message,
                 "reference": {**log_reference, "csv_row": int(row.csv_row)}}
                for row in chosen.itertuples(index=False)]
    return {"as_of": origin.isoformat(), "lookback_minutes": lookback_minutes,
            "monitoring_window_start": history.timestamp.iloc[0].isoformat(),
            "monitoring_rows": len(history), "monitoring": metric_evidence,
            "monitoring_reference": monitoring_reference, "recent_log_count": len(earlier),
            "returned_log_count": len(messages), "logs_most_recent_first": messages,
            "log_source": log_reference,
            "limitations": ["Monitoring baselines are descriptive medians, not calibrated alert thresholds.",
                            "Recent log messages are context, not proof of a metric change or root cause."]}


def validate_query(question: str) -> str:
    if not isinstance(question, str) or not question.strip() or len(question) > 4096:
        raise ValueError("Supply an investigation question of at most 4096 characters.")
    return question.strip()


def assemble_investigation(data_dir: Path, *, question: str, as_of: str,
                           index_dir: Path, log_model_dir: Path, anomaly_model_dir: Path,
                           lookback_minutes: int = 30, max_results: int = 3,
                           include_incident_snapshots: bool = False) -> dict:
    """Call previously fitted models. Do not retrain or generate a diagnosis."""
    from .anomaly_detection import MonitoringDetector
    from .knowledge_retrieval import KnowledgeStore
    from .log_clustering import LogClassifier

    question = validate_query(question)
    packet = read_evidence(data_dir, as_of=as_of, lookback_minutes=lookback_minutes)
    store = KnowledgeStore(index_dir)
    if store.source_changes(Path(data_dir)):
        raise ValueError("Knowledge sources have changed since indexing; rebuild the local index.")
    if naive_time(packet["as_of"]) <= store.cutoff:
        raise ValueError("as_of must be after the knowledge index training cutoff.")

    # Keep both model evaluations time-bounded by their existing inference contracts.
    log_classifier = LogClassifier(log_model_dir)
    for row in packet["logs_most_recent_first"]:
        row["classification"] = log_classifier.predict(row["log_message"], row["log_level"])
    monitoring = pd.read_csv(Path(data_dir) / MONITORING_FILE)
    score = MonitoringDetector(anomaly_model_dir).predict(monitoring, as_of=packet["as_of"])
    if score["timestamp"] != packet["as_of"]:
        raise ValueError("Monitoring detector did not score the requested origin.")
    results = store.search(question, top_k=max_results, source_kind="runbooks", as_of=packet["as_of"])
    incidents = None
    if include_incident_snapshots:
        incidents = store.search(question, top_k=max_results,
                                 source_kind="incident_snapshot", as_of=packet["as_of"])
    return {"scope": "time_bounded_incident_evidence", "question": question,
            "observations": packet, "monitoring_detector": score,
            "runbook_retrieval": results, "historical_incident_retrieval": incidents,
            "interpretation": {
                "status": "evidence_only",
                "root_cause": None,
                "resolution": None,
                "notes": ["A monitoring review flag is not a diagnosis.",
                          "A log category is an exact catalog mapping, not a learned cause.",
                          "Retrieved passages describe possibilities and source text, not verified causes.",
                          "The incident triage model is not applied to live monitoring rows because that transfer is unvalidated."]}}
