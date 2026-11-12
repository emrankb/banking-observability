"""Contextual monitoring outliers with an explicit, held-out review threshold."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits

from .detection_common import MONITORING_METRICS, load_detector, metric_frame

WINDOW = 60
LOAD_METRICS = ("cpu_utilization", "memory_usage", "db_connection_pool_usage", "network_throughput_mbps",
                "avg_transaction_processing_time_ms", "transaction_volume_per_hour")
RATE_METRICS = ("neft_success_rate", "imps_success_rate", "rtgs_success_rate", "auth_success_rate")
CANDIDATES = tuple({"n_estimators": trees, "max_samples": samples}
                   for trees, samples in product((128, 256), (128, 512)))


def anomaly_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = metric_frame(history, MONITORING_METRICS, step_minutes=1).set_index("timestamp")
    features = {}
    for col in LOAD_METRICS:
        # Exclude the current reading from its own reference window.
        reference = frame[col].shift(1).rolling(WINDOW, min_periods=WINDOW).mean()
        features[f"{col}__minus_prior_60_mean"] = frame[col] - reference
    for col in RATE_METRICS:
        features[f"{col}__non_success_pct"] = 100.0 - frame[col]
    features["application_error_rate"] = frame.application_error_rate
    features["service_uptime__loss_pct"] = 100.0 - frame.service_uptime
    return pd.DataFrame(features, index=frame.index).dropna()


def review_threshold(scores, fraction: float) -> float:
    if not np.isfinite(fraction) or not 0 < fraction < 0.5:
        raise ValueError("review_fraction must be between 0 and 0.5, exclusive.")
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Threshold calibration requires at least two finite scores.")
    # Strict greater-than at inference; tied scores do not force a quota of alerts.
    return float(np.quantile(scores, 1 - fraction, method="higher"))


def forest(spec: dict, seed: int) -> IsolationForest:
    return IsolationForest(**spec, contamination="auto", max_features=1.0,
                           bootstrap=False, random_state=seed, n_jobs=1)


def rank_stability(score_vectors: list[np.ndarray], fraction: float) -> tuple[float, float]:
    k = max(1, int(np.ceil(len(score_vectors[0]) * fraction)))
    ranked_sets = [set(np.argsort(-values, kind="stable")[:k]) for values in score_vectors]
    jaccard, rank_correlations = [], []
    for a, b in combinations(range(len(score_vectors)), 2):
        jaccard.append(len(ranked_sets[a] & ranked_sets[b]) / len(ranked_sets[a] | ranked_sets[b]))
        rank_correlations.append(float(spearmanr(score_vectors[a], score_vectors[b]).statistic))
    if not np.isfinite(rank_correlations).all():
        raise ValueError("Constant outlier scores prevent candidate stability comparison.")
    return float(np.mean(jaccard)), float(np.mean(rank_correlations))


@dataclass
class AnomalyModel:
    forest: IsolationForest
    columns: list[str]
    specification: dict
    threshold: float
    robust_threshold: float
    review_fraction: float
    reference_median: pd.Series
    reference_scale: pd.Series
    constant_values: dict
    fit_through: str
    calibrated_through: str

    def raw_scores(self, features: pd.DataFrame) -> np.ndarray:
        if features.columns.tolist() != self.columns or not np.isfinite(features.to_numpy()).all():
            raise ValueError("Anomaly feature schema or values are invalid.")
        with threadpool_limits(limits=1):
            return -self.forest.score_samples(features)

    def reference_distances(self, features: pd.DataFrame) -> pd.DataFrame:
        return (features - self.reference_median).abs().div(self.reference_scale)

    def constant_changes(self, features: pd.DataFrame) -> list[list[str]]:
        return [[name for name, value in self.constant_values.items()
                 if not np.isclose(row[name], value, rtol=0, atol=1e-10)] for _, row in features.iterrows()]

    def assess(self, features: pd.DataFrame) -> pd.DataFrame:
        scores = self.raw_scores(features)
        distances = self.reference_distances(features)
        constants = self.constant_changes(features)
        flagged = scores > self.threshold
        changed = np.array([bool(values) for values in constants])
        return pd.DataFrame({"timestamp": features.index, "anomaly_score": scores, "threshold": self.threshold,
                             "forest_flag": flagged, "constant_feature_change": changed,
                             "review_required": flagged | changed,
                             "constant_features_changed": [";".join(values) for values in constants],
                             "robust_score": distances.max(axis=1).to_numpy(), "robust_threshold": self.robust_threshold,
                             "robust_flag": (distances.max(axis=1).to_numpy() > self.robust_threshold) | changed})


def fit_anomaly(train: pd.DataFrame, seed: int, review_fraction: float = 0.01):
    clean = metric_frame(train, MONITORING_METRICS, step_minutes=1)
    if len(clean) < 3 * 1440:
        raise ValueError("Need at least three days to separate fit, selection, and calibration periods.")
    review_threshold([0, 1], review_fraction)
    features = anomaly_features(clean)
    calibration_start = clean.timestamp.iloc[-1440]
    selection_start = clean.timestamp.iloc[-2880]
    initial_fit = features.loc[features.index < selection_start]
    selection = features.loc[(features.index >= selection_start) & (features.index < calibration_start)]
    calibration = features.loc[features.index >= calibration_start]
    stability_rows = []
    for spec in CANDIDATES:
        vectors = []
        for offset in (0, 11, 29):
            model = forest(spec, (seed + offset) % (2**32))
            with threadpool_limits(limits=1):
                model.fit(initial_fit)
                vectors.append(-model.score_samples(selection))
        overlap, correlation = rank_stability(vectors, review_fraction)
        stability_rows.append({**spec, "top_rank_jaccard": overlap, "score_spearman": correlation})
    stability = pd.DataFrame(stability_rows).sort_values(
        ["top_rank_jaccard", "score_spearman", "n_estimators", "max_samples"], ascending=[False, False, True, True]).reset_index(drop=True)
    chosen = {name: int(stability.iloc[0][name]) for name in ("n_estimators", "max_samples")}
    final_fit = features.loc[features.index < calibration_start]
    final = forest(chosen, seed)
    with threadpool_limits(limits=1):
        final.fit(final_fit)
        cal_scores = -final.score_samples(calibration)
    median = final_fit.median()
    iqr = final_fit.quantile(0.75) - final_fit.quantile(0.25)
    # Constants get a separate guard; a one-unit denominator is only for displaying deviations.
    scale = iqr.where(iqr > 0, final_fit.std(ddof=0)).replace(0, 1.0)
    constants = {col: float(final_fit[col].iloc[0]) for col in final_fit if final_fit[col].nunique() == 1}
    robust_cal = calibration.sub(median).abs().div(scale).max(axis=1)
    fitted = AnomalyModel(final, features.columns.tolist(), chosen, review_threshold(cal_scores, review_fraction),
                          review_threshold(robust_cal, review_fraction), review_fraction, median, scale, constants,
                          str(final_fit.index[-1]), str(calibration.index[-1]))
    plan = {"candidate_parameters": list(CANDIDATES), "selected_parameters": chosen, "seed": seed, "seed_offsets": [0, 11, 29],
            "selection_rule": "Maximum mean top-rank Jaccard across seeds, then score-rank Spearman; ties prefer fewer trees/samples.",
            "selection_meaning": "Ranking repeatability on later unlabeled readings, not measured incident-detection accuracy.",
            "review_fraction": review_fraction, "threshold": fitted.threshold, "robust_threshold": fitted.robust_threshold,
            "initial_fit_rows": len(initial_fit), "initial_fit_through": str(initial_fit.index[-1]),
            "selection_start": str(selection.index[0]), "selection_end": str(selection.index[-1]), "selection_rows": len(selection),
            "final_fit_rows": len(final_fit), "final_fit_through": fitted.fit_through,
            "calibration_start": str(calibration.index[0]), "calibrated_through": fitted.calibrated_through,
            "calibration_rows": len(calibration), "columns": fitted.columns, "constant_fit_features": constants,
            "threshold_rule": "Score strictly exceeds the higher empirical (1 - review_fraction) calibration quantile.",
            "contamination": "auto", "uses_predict_or_decision_function": False,
            "scope": "Unusual current metric behavior; review flags are neither confirmed failures nor forecasts."}
    return fitted, plan, {"anomaly_stability": stability, "anomaly_calibration": fitted.assess(calibration)}


def alert_summary(scored: pd.DataFrame) -> dict:
    alerts = scored.loc[scored.review_required]
    episode_count = int(alerts.timestamp.diff().ne(pd.Timedelta(minutes=1)).sum()) if len(alerts) else 0
    return {"rows": len(scored), "forest_flagged_minutes": int(scored.forest_flag.sum()),
            "constant_change_minutes": int(scored.constant_feature_change.sum()),
            "review_minutes": int(scored.review_required.sum()), "review_fraction_observed": float(scored.review_required.mean()),
            "consecutive_review_episodes": episode_count, "robust_baseline_review_minutes": int(scored.robust_flag.sum())}


class MonitoringDetector:
    def __init__(self, directory: Path):
        self.model, self.metadata = load_detector(directory, kind="monitoring_anomaly", expected_type=AnomalyModel)

    def predict(self, history: pd.DataFrame, *, as_of: str | None = None) -> dict:
        if "timestamp" not in history or history.empty:
            raise ValueError("Supply timestamped monitoring history.")
        times = pd.to_datetime(history.timestamp, errors="raise")
        if not pd.api.types.is_datetime64_any_dtype(times) or times.isna().any() or times.dt.tz is not None:
            raise ValueError("Use the non-null timezone-naive monitoring clock.")
        origin = pd.Timestamp(as_of) if as_of is not None else times.iloc[-1]
        if pd.isna(origin) or origin.tzinfo is not None or not times.eq(origin).any():
            raise ValueError("as_of must match an observed timezone-naive timestamp.")
        if origin <= pd.Timestamp(self.model.calibrated_through):
            raise ValueError("Scoring origin must be after threshold calibration; use the training report for earlier readings.")
        prefix = metric_frame(history.loc[times.le(origin)], MONITORING_METRICS, step_minutes=1)
        if len(prefix) < WINDOW + 1:
            raise ValueError("Need 61 consecutive readings: the current minute and its preceding 60 minutes.")
        recent = prefix.tail(WINDOW + 1)
        features = anomaly_features(recent)
        assessed = self.model.assess(features).iloc[-1]
        distances = self.model.reference_distances(features).iloc[-1].sort_values(ascending=False)
        evidence = [{"feature": col, "value": float(features.iloc[-1][col]),
                     "fit_median": float(self.model.reference_median[col]),
                     "reference_scale": float(self.model.reference_scale[col]),
                     "absolute_scaled_deviation": float(distances[col])} for col in distances.index[:3]]
        return {"timestamp": origin.isoformat(), "anomaly_score": float(assessed.anomaly_score),
                "threshold": self.model.threshold, "forest_flag": bool(assessed.forest_flag),
                "constant_features_changed": self.model.constant_changes(features)[0],
                "review_required": bool(assessed.review_required), "history_rows_used": len(recent),
                "largest_reference_deviations": evidence, "deviations_are_forest_attributions": False,
                "observed_metrics": {col: float(recent.iloc[-1][col]) for col in MONITORING_METRICS},
                "model_sha256": self.metadata["model_sha256"], "calibrated_through": self.model.calibrated_through,
                "scope": "current_monitoring_review", "source_timezone": "unspecified; timezone-naive timestamps",
                "limitation": "Review flag only. No labeled monitoring incidents establish precision, recall, or root cause."}
