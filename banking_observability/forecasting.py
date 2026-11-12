"""Load the trained artifact and forecast with the same feature code as training."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .forecast_features import FeatureSpec, REQUIRED_COLUMNS, TARGET, build_features, validate_history
from .forecast_models import FittedCandidate

RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}


class LatencyForecaster:
    """Only load artifacts you trust. A checksum detects damage, not malicious provenance."""

    def __init__(self, directory: Path):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata.get("format_version") != 1:
            raise ValueError("Unsupported forecast artifact format.")
        artifact = directory / "model.joblib"
        if file_sha256(artifact) != self.metadata["model_sha256"]:
            raise ValueError("Model checksum does not match metadata; artifact loading refused.")
        versions = runtime_versions()
        mismatches = {k: (expected, versions.get(k)) for k, expected in self.metadata["runtime_versions"].items() if versions.get(k) != expected}
        if mismatches:
            raise ValueError(f"Artifact/runtime version mismatch: {mismatches}. Use requirements-forecast.txt or retrain.")
        feature_module = Path(__file__).with_name("forecast_features.py")
        if file_sha256(feature_module) != self.metadata["feature_code_sha256"]:
            raise ValueError("Feature code differs from the trained artifact. Retrain before inference.")
        self.spec = FeatureSpec.from_dict(self.metadata["feature_spec"])
        self.model = joblib.load(artifact)
        if not isinstance(self.model, FittedCandidate):
            raise ValueError("Artifact is not a fitted latency candidate.")
        if self.model.specification != self.metadata["candidate"]:
            raise ValueError("Artifact candidate does not match its metadata.")
        if self.model.columns != self.metadata["selected_feature_columns"]:
            raise ValueError("Artifact feature order does not match its metadata.")

    def predict(self, history: pd.DataFrame, *, as_of: str | pd.Timestamp | None = None) -> dict:
        """Ignore observations after as_of; require complete causal lookback through it."""
        if "timestamp" not in history:
            raise ValueError("Missing required columns: timestamp")
        times = pd.to_datetime(history["timestamp"], errors="raise")
        if times.empty:
            raise ValueError("History is empty.")
        origin = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(times.iloc[-1])
        if origin.tzinfo is not None or pd.isna(origin):
            raise ValueError("Forecast origin must use the timezone-naive source clock.")
        if not times.eq(origin).any():
            raise ValueError("Forecast origin must exactly match an observed monitoring timestamp.")
        if origin <= pd.Timestamp(self.metadata["trained_target_through"]):
            raise ValueError("Origin is not after the fitted model's training targets; retrospective scoring would leak.")
        # A full replay file may be supplied. Later observations are removed BEFORE
        # feature construction, and therefore cannot change this forecast.
        causal = history.loc[times.le(origin)]
        causal = validate_history(causal, minimum_rows=self.spec.minimum_history_rows)
        causal = causal.tail(self.spec.minimum_history_rows)
        features = build_features(causal, self.spec).tail(1)
        if features.index[-1] != origin or features.isna().any().any():
            raise ValueError("Insufficient aligned feature history for forecast origin.")
        if features.columns.tolist() != self.metadata["all_feature_columns"]:
            raise ValueError("Feature schema differs from trained schema.")
        with threadpool_limits(limits=1):
            prediction = float(self.model.predict(features)[0])
        if not np.isfinite(prediction):
            raise RuntimeError("Non-finite forecast.")
        last = causal.iloc[-1]
        outside = []
        for column in REQUIRED_COLUMNS[1:]:
            bounds = self.metadata["training_metric_ranges"][column]
            if not bounds["min"] <= float(last[column]) <= bounds["max"]:
                outside.append(column)
        return {
            "forecast_origin": origin.isoformat(),
            "target_timestamp": (origin + pd.Timedelta(minutes=self.spec.horizon_minutes)).isoformat(),
            "horizon_minutes": self.spec.horizon_minutes,
            "target_column": TARGET,
            "prediction_ms": prediction,
            "last_observed_ms": float(last[TARGET]),
            "daily_seasonal_naive_ms": float(features["latency__previous_day_at_target"].iloc[0]),
            "history_rows_used": len(causal),
            "model_name": self.metadata["candidate"]["name"],
            "model_sha256": self.metadata["model_sha256"],
            "trained_target_through": self.metadata["trained_target_through"],
            "latest_metrics_outside_training_range": outside,
            "source_timezone": "unspecified; timezone-naive timestamps",
        }
