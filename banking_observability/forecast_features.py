"""Causal minute-grid features for a direct five-minute latency forecast.

Rows are indexed by forecast origin t. Observations through t are assumed
available after the t reading has arrived. The target is the single reading at
t + horizon, not the mean over the next horizon. No future resource measurement
is an input. Calendar fields for the target time are knowable at t.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "avg_transaction_processing_time_ms"
RESOURCE_COLUMNS = (
    "cpu_utilization", "memory_usage", "db_connection_pool_usage",
    "network_throughput_mbps", "transaction_volume_per_hour",
)
REQUIRED_COLUMNS = ("timestamp", TARGET, *RESOURCE_COLUMNS)


@dataclass(frozen=True)
class FeatureSpec:
    horizon_minutes: int = 5
    latency_lags: tuple[int, ...] = (0, 1, 5, 15, 60, 1440)
    resource_lags: tuple[int, ...] = (0, 5, 15)
    rolling_windows: tuple[int, ...] = (5, 15, 60)
    seasonal_period_minutes: int = 1440

    def __post_init__(self) -> None:
        if not 0 < self.horizon_minutes < self.seasonal_period_minutes:
            raise ValueError("Horizon must be positive and shorter than the daily seasonal period.")
        if not self.latency_lags or not self.resource_lags or not self.rolling_windows:
            raise ValueError("Lag and rolling-window definitions cannot be empty.")
        if min(*self.latency_lags, *self.resource_lags) < 0 or min(self.rolling_windows) < 1:
            raise ValueError("Lags must be nonnegative and rolling windows positive.")

    @property
    def minimum_history_rows(self) -> int:
        return max(
            *self.latency_lags, *self.resource_lags,
            *(w - 1 for w in self.rolling_windows),
            self.seasonal_period_minutes - self.horizon_minutes,
        ) + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "FeatureSpec":
        value = value.copy()
        for name in ("latency_lags", "resource_lags", "rolling_windows"):
            value[name] = tuple(value[name])
        return cls(**value)


def validate_history(frame: pd.DataFrame, *, minimum_rows: int = 1) -> pd.DataFrame:
    """Reject incomplete or ambiguous history; do not repair it silently."""
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate input column names are not allowed.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if len(frame) < minimum_rows:
        raise ValueError(f"Need at least {minimum_rows} history rows; received {len(frame)}.")
    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid timestamp in monitoring history.") from exc
    time = result["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(time.dtype):
        raise ValueError("Timestamps must use one consistent timezone-naive source clock.")
    if time.isna().any() or time.dt.tz is not None:
        raise ValueError("Timestamps must be non-null and timezone-naive, matching the source contract.")
    if time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Timestamps must be strictly increasing with no duplicates.")
    if not time.eq(time.dt.floor("min")).all():
        raise ValueError("Monitoring timestamps must lie on minute boundaries.")
    if not time.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("History must be a complete one-minute grid; missing minutes are not imputed.")
    numeric = result.columns.drop("timestamp")
    try:
        result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("All required metric values must be numeric.") from exc
    if not np.isfinite(result[numeric].to_numpy()).all():
        raise ValueError("Required metrics must be finite; nulls and infinity are not imputed.")
    if (result[numeric] < 0).any().any():
        raise ValueError("Required resource/processing metrics cannot be negative.")
    for column in ("cpu_utilization", "memory_usage", "db_connection_pool_usage"):
        if result[column].gt(100).any():
            raise ValueError(f"{column} exceeds this model's 0–100 monitoring contract; check measurement scope.")
    return result.reset_index(drop=True)


def read_history(path: Path, *, minimum_rows: int = 1) -> pd.DataFrame:
    return validate_history(pd.read_csv(path), minimum_rows=minimum_rows)


def build_features(history: pd.DataFrame, spec: FeatureSpec = FeatureSpec()) -> pd.DataFrame:
    """One row per origin, with leading warmup NaNs but no target column.

    An inclusive rolling window is intentional: at origin t, y[t] is observed.
    Shifting every window by one would needlessly discard the newest reading.
    Future targets are constructed separately in make_supervised().
    """
    frame = validate_history(history)
    index = pd.DatetimeIndex(frame.timestamp, name="forecast_origin")
    values = frame.set_index(index)
    target_time = index + pd.Timedelta(minutes=spec.horizon_minutes)
    minute = target_time.hour * 60 + target_time.minute
    data: dict[str, object] = {
        "clock__minute_of_day": minute,
        "clock__hour": target_time.hour,
        "clock__quarter_hour": minute // 15,
        "clock__is_weekend": (target_time.dayofweek >= 5).astype(int),
    }
    for harmonic in (1, 2, 3):
        angle = 2 * np.pi * harmonic * minute / spec.seasonal_period_minutes
        data[f"clock__sin_{harmonic}"] = np.sin(angle)
        data[f"clock__cos_{harmonic}"] = np.cos(angle)
    latency = values[TARGET]
    for lag in spec.latency_lags:
        data[f"latency__lag_{lag}"] = latency.shift(lag)
    for window in spec.rolling_windows:
        roll = latency.rolling(window, min_periods=window)
        data[f"latency__mean_{window}"] = roll.mean()
        data[f"latency__std_{window}"] = roll.std(ddof=0)
    data["latency__change_5"] = latency - latency.shift(5)
    # For y[t+5], daily persistence is y[t+5-1440], not y[t-1440].
    data["latency__previous_day_at_target"] = latency.shift(
        spec.seasonal_period_minutes - spec.horizon_minutes
    )
    for column in RESOURCE_COLUMNS:
        series = values[column]
        prefix = f"resource__{column}"
        for lag in spec.resource_lags:
            data[f"{prefix}__lag_{lag}"] = series.shift(lag)
        for window in spec.rolling_windows:
            data[f"{prefix}__mean_{window}"] = series.rolling(window, min_periods=window).mean()
        data[f"{prefix}__std_15"] = series.rolling(15, min_periods=15).std(ddof=0)
        data[f"{prefix}__change_5"] = series - series.shift(5)
    return pd.DataFrame(data, index=index, dtype="float64")


def feature_columns(features: pd.DataFrame, group: str) -> list[str]:
    if group == "clock":
        return [c for c in features if c.startswith("clock__")]
    if group == "latency_clock":
        return [c for c in features if not c.startswith("resource__")]
    if group == "all":
        return features.columns.tolist()
    raise ValueError(f"Unknown feature group: {group}")


def make_supervised(history: pd.DataFrame, spec: FeatureSpec = FeatureSpec()) -> tuple[pd.DataFrame, pd.Series]:
    """Remove only explicit lag warmup and unknown future target rows."""
    frame = validate_history(history)
    features = build_features(frame, spec)
    target = pd.Series(
        frame[TARGET].shift(-spec.horizon_minutes).to_numpy(),
        index=features.index, name="target_ms",
    )
    valid = features.notna().all(axis=1) & target.notna()
    return features.loc[valid], target.loc[valid]
