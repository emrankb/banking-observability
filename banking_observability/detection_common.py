"""Input checks and artifact storage shared by incident and anomaly models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .forecasting import file_sha256, runtime_versions

INCIDENT_METRICS = (
    "cpu_utilization", "memory_usage", "db_connection_pool_usage",
    "network_throughput_mbps", "application_error_rate", "neft_success_rate",
    "imps_success_rate", "rtgs_success_rate", "avg_transaction_processing_time_ms",
)
MONITORING_METRICS = (*INCIDENT_METRICS, "auth_success_rate", "transaction_volume_per_hour", "service_uptime")
PERCENTAGES = {"cpu_utilization", "memory_usage", "db_connection_pool_usage", "neft_success_rate",
               "imps_success_rate", "rtgs_success_rate", "auth_success_rate", "service_uptime"}
DISPUTED_METRICS = ("cpu_utilization", "memory_usage", "db_connection_pool_usage", "application_error_rate", "neft_success_rate")


def metric_frame(frame: pd.DataFrame, columns: tuple[str, ...], *, step_minutes: int | None = None) -> pd.DataFrame:
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Supply nonempty data with unique column names.")
    required = ("timestamp", *columns)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    result = frame.loc[:, required].copy()
    try:
        time = pd.to_datetime(result.timestamp, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid timestamp.") from exc
    if not pd.api.types.is_datetime64_any_dtype(time) or time.isna().any() or time.dt.tz is not None:
        raise ValueError("Use non-null timezone-naive timestamps, matching the source clock.")
    if time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Timestamps must be strictly increasing, without duplicates.")
    if not time.eq(time.dt.floor("min")).all():
        raise ValueError("Timestamps must be on minute boundaries.")
    if step_minutes is not None and not time.diff().iloc[1:].eq(pd.Timedelta(minutes=step_minutes)).all():
        raise ValueError(f"Expected a complete {step_minutes}-minute grid; missing readings are not imputed.")
    result["timestamp"] = time
    try:
        result[list(columns)] = result[list(columns)].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Metrics must be numeric.") from exc
    values = result[list(columns)].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Metrics must be finite and nonnegative.")
    for col in set(columns) & PERCENTAGES:
        if result[col].gt(100).any():
            raise ValueError(f"{col} exceeds the 0–100 monitoring contract; check measurement scope.")
    for col in set(columns) & {"application_error_rate", "transaction_volume_per_hour"}:
        if not np.equal(result[col], np.floor(result[col])).all():
            raise ValueError(f"{col} must contain whole values, as in the supplied schema.")
    return result.reset_index(drop=True)


def read_incidents(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    result = metric_frame(raw, INCIDENT_METRICS, step_minutes=60)
    if not {"incident_id", "is_critical"}.issubset(raw):
        raise ValueError("Incident data require incident_id and is_critical.")
    if raw.incident_id.isna().any() or raw.incident_id.duplicated().any():
        raise ValueError("Incident IDs must be non-null and unique.")
    if not raw.is_critical.isin([0, 1]).all():
        raise ValueError("is_critical must contain the supplied binary labels 0 or 1.")
    result.insert(0, "incident_id", raw.incident_id.astype(str))
    result["is_critical"] = raw.is_critical.astype(int)
    return result


def deterministic_seed(path: Path) -> int:
    # File-derived reproducibility seed, independent of validation performance.
    return int(file_sha256(path)[:8], 16)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")


def source_hashes(module: str) -> dict[str, str]:
    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in ("detection_common.py", module)}


def save_detector(directory: Path, detector: object, contract: dict, *, module: str) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    contract = {**contract, "format_version": 1, "runtime_versions": runtime_versions(), "code_sha256": source_hashes(module)}
    model_file = directory / "model.joblib"
    joblib.dump({"detector": detector, "contract": contract}, model_file, compress=3)
    metadata = {**contract, "model_sha256": file_sha256(model_file)}
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_detector(directory: Path, *, kind: str, expected_type: type):
    """Checks consistency, not provenance. Only open trusted local joblib artifacts."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("format_version") != 1 or metadata.get("kind") != kind:
        raise ValueError("Unsupported detector artifact format or kind.")
    model_file = directory / "model.joblib"
    if file_sha256(model_file) != metadata.get("model_sha256"):
        raise ValueError("Detector checksum mismatch; loading refused.")
    if metadata.get("runtime_versions") != runtime_versions():
        raise ValueError("Detector/runtime version mismatch; use requirements-forecast.txt or retrain.")
    for name, sha in metadata["code_sha256"].items():
        if Path(name).name != name or file_sha256(Path(__file__).with_name(name)) != sha:
            raise ValueError("Detector code differs from the saved artifact; retrain.")
    payload = joblib.load(model_file)
    contract = {key: value for key, value in metadata.items() if key != "model_sha256"}
    if not isinstance(payload, dict) or payload.get("contract") != contract or not isinstance(payload.get("detector"), expected_type):
        raise ValueError("Detector payload and metadata disagree.")
    return payload["detector"], metadata


def range_evidence(row: pd.Series, ranges: dict) -> list[dict]:
    return [{"metric": col, "value": float(row[col]), "fit_min": limits["min"], "fit_max": limits["max"]}
            for col, limits in ranges.items() if not limits["min"] <= float(row[col]) <= limits["max"]]


def metric_ranges(frame: pd.DataFrame, columns) -> dict:
    return {col: {"min": float(frame[col].min()), "max": float(frame[col].max())} for col in columns}
