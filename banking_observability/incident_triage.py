"""Binary triage of the supplied incident snapshots, not future incidents."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score, precision_score, recall_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .detection_common import (DISPUTED_METRICS, INCIDENT_METRICS, load_detector, metric_frame,
                               metric_ranges, range_evidence)

THRESHOLDS = (0.2, 0.35, 0.5, 0.65, 0.8)
CANDIDATES = tuple(
    [{"name": f"weighted_logistic_c_{c:g}", "kind": "logistic", "C": c} for c in (0.1, 1.0, 10.0)]
    + [{"name": f"weighted_tree_depth_{depth}", "kind": "tree", "max_depth": depth} for depth in (1, 2)]
)


def estimator(spec: dict, seed: int):
    if spec["kind"] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=spec["C"], l1_ratio=0, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-8))
    if spec["kind"] == "tree":
        return DecisionTreeClassifier(max_depth=spec["max_depth"], min_samples_leaf=5,
                                      class_weight="balanced", random_state=seed)
    raise ValueError(f"Unknown triage candidate: {spec}")


def classification_metrics(y, scores, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "critical_count": int(y.sum()), "prevalence": float(y.mean()),
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
            "average_precision": float(average_precision_score(y, scores)) if np.unique(y).size == 2 else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


@dataclass
class TriageModel:
    estimator: object
    columns: list[str]
    specification: dict
    threshold: float
    fit_through: str
    ranges: dict

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        clean = metric_frame(frame, INCIDENT_METRICS)
        with threadpool_limits(limits=1):
            return self.estimator.predict_proba(clean[self.columns])[:, 1]


def fit_triage(development: pd.DataFrame, seed: int) -> tuple[TriageModel, dict, dict[str, pd.DataFrame]]:
    if len(development) < 110 or development.is_critical.nunique() != 2:
        raise ValueError("Triage selection needs at least 110 hourly records and both classes.")
    folds = list(TimeSeriesSplit(n_splits=3, test_size=24, gap=1).split(development))
    columns = list(INCIDENT_METRICS)
    fold_rows, predictions, split_rows = [], [], []
    for fold, (fit_idx, validation_idx) in enumerate(folds, 1):
        fit = development.iloc[fit_idx]; validation = development.iloc[validation_idx]
        if fit.is_critical.nunique() != 2 or validation.is_critical.nunique() != 2:
            raise ValueError("Each chronological training/validation block must contain both classes.")
        counts = fit.is_critical.value_counts().to_dict()
        split_rows.append({"fold": fold, "fit_start": str(fit.timestamp.iloc[0]), "fit_end": str(fit.timestamp.iloc[-1]),
                           "validation_start": str(validation.timestamp.iloc[0]), "validation_end": str(validation.timestamp.iloc[-1]),
                           "fit_rows": len(fit), "fit_critical": int(fit.is_critical.sum()),
                           "validation_rows": len(validation), "validation_critical": int(validation.is_critical.sum()),
                           "noncritical_weight": len(fit) / (2 * counts[0]), "critical_weight": len(fit) / (2 * counts[1])})
        for order, spec in enumerate(CANDIDATES):
            model = estimator(spec, seed)
            with threadpool_limits(limits=1):
                model.fit(fit[columns], fit.is_critical)
                scores = model.predict_proba(validation[columns])[:, 1]
            predictions.append(pd.DataFrame({"candidate": spec["name"], "fold": fold,
                                              "timestamp": validation.timestamp.to_numpy(),
                                              "actual": validation.is_critical.to_numpy(), "critical_score": scores}))
            for threshold in THRESHOLDS:
                fold_rows.append({"candidate": spec["name"], "order": order, "fold": fold, "threshold": threshold,
                                  **classification_metrics(validation.is_critical, scores, threshold)})
    fold_table = pd.DataFrame(fold_rows)
    leaderboard = fold_table.groupby(["candidate", "order", "threshold"], as_index=False)[["f2", "f1", "precision", "recall", "average_precision"]].mean()
    leaderboard["distance_from_half"] = (leaderboard.threshold - 0.5).abs()
    leaderboard = leaderboard.sort_values(["f2", "average_precision", "precision", "order", "distance_from_half", "threshold"],
                                           ascending=[False, False, False, True, True, True]).reset_index(drop=True)
    winner = leaderboard.iloc[0]
    chosen = next(spec for spec in CANDIDATES if spec["name"] == winner.candidate)
    final = estimator(chosen, seed)
    with threadpool_limits(limits=1):
        final.fit(development[columns], development.is_critical)
    fitted = TriageModel(final, columns, chosen, float(winner.threshold), str(development.timestamp.iloc[-1]), metric_ranges(development, columns))
    selection = {"candidate": chosen, "threshold": fitted.threshold, "columns": columns, "seed": seed,
                 "fit_through": fitted.fit_through, "fit_rows": len(development), "fit_critical": int(development.is_critical.sum()),
                 "selection_rule": "Mean chronological F2, then average precision, precision, candidate order, and threshold closest to 0.5.",
                 "thresholds": list(THRESHOLDS), "candidates": list(CANDIDATES),
                 "scope": "Current-state classification of supplied incident snapshots. Scores are uncalibrated; not future-incident risk."}
    return fitted, selection, {"triage_cv_metrics": fold_table, "triage_cv_leaderboard": leaderboard,
                               "triage_cv_predictions": pd.concat(predictions, ignore_index=True), "triage_folds": pd.DataFrame(split_rows)}


def audit_incidents(incidents: pd.DataFrame, monitoring: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    aligned = incidents.merge(monitoring[["timestamp", *INCIDENT_METRICS]], on="timestamp", how="left",
                             suffixes=("_incident", "_monitor"), validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Incident audit requires an exact monitoring timestamp for every incident.")
    cells = []
    for col in INCIDENT_METRICS:
        a, b = aligned[col + "_incident"], aligned[col + "_monitor"]
        different = ~np.isclose(a, b, rtol=0, atol=1e-8)
        for i in np.flatnonzero(different):
            cells.append({"incident_id": str(aligned.iloc[i].incident_id), "timestamp": str(aligned.iloc[i].timestamp),
                          "is_critical": int(aligned.iloc[i].is_critical), "metric": col,
                          "incident_value": float(a.iloc[i]), "monitoring_value": float(b.iloc[i]), "difference": float(a.iloc[i] - b.iloc[i])})
    differences = pd.DataFrame(cells, columns=["incident_id", "timestamp", "is_critical", "metric", "incident_value", "monitoring_value", "difference"])
    rule = incidents.cpu_utilization.gt(85).astype(int)
    summary = {"incident_rows": len(incidents), "critical_rows": int(incidents.is_critical.sum()),
               "disagreeing_rows": int(differences.incident_id.nunique()), "disagreeing_cells": len(differences),
               "disagreeing_metrics": sorted(differences.metric.unique().tolist()),
               "cpu_above_85_label_agreements": int(rule.eq(incidents.is_critical).sum()),
               "note": "The CPU rule is a full-file provenance diagnostic discovered during EDA, not an independently evaluated model."}
    return summary, differences


def evaluate_triage(fitted: TriageModel, development: pd.DataFrame, evaluation: pd.DataFrame, monitoring: pd.DataFrame, seed: int):
    predictions, metric_rows = [], []
    scores = fitted.score(evaluation)
    same_times = evaluation[["timestamp"]].merge(monitoring, on="timestamp", how="left", validate="one_to_one")
    comparisons = [("selected_incident_model", scores, fitted.threshold),
                   ("always_noncritical", np.zeros(len(evaluation)), 0.5),
                   ("same_model_monitoring_snapshots", fitted.score(same_times), fitted.threshold)]
    # Sensitivity only: remove the five fields that conflict across sources.
    remaining = [col for col in INCIDENT_METRICS if col not in DISPUTED_METRICS]
    ablated = estimator(fitted.specification, seed)
    with threadpool_limits(limits=1):
        ablated.fit(development[remaining], development.is_critical)
        ablated_scores = ablated.predict_proba(evaluation[remaining])[:, 1]
    comparisons.append(("disputed_fields_removed_sensitivity", ablated_scores, fitted.threshold))
    for name, values, threshold in comparisons:
        metric_rows.append({"comparison": name, "threshold": threshold, **classification_metrics(evaluation.is_critical, values, threshold)})
        predictions.append(pd.DataFrame({"comparison": name, "incident_id": evaluation.incident_id.to_numpy(),
                                        "timestamp": evaluation.timestamp.to_numpy(), "actual": evaluation.is_critical.to_numpy(),
                                        "critical_score": values, "threshold": threshold, "predicted": (values >= threshold).astype(int)}))
    return pd.DataFrame(metric_rows), pd.concat(predictions, ignore_index=True)


class IncidentTriage:
    def __init__(self, directory: Path):
        self.model, self.metadata = load_detector(directory, kind="incident_triage", expected_type=TriageModel)

    def predict(self, incident: dict) -> dict:
        frame = metric_frame(pd.DataFrame([incident]), INCIDENT_METRICS)
        score = float(self.model.score(frame)[0])
        return {"incident_id": incident.get("incident_id"), "timestamp": frame.timestamp.iloc[0].isoformat(),
                "critical_score": score, "threshold": self.model.threshold, "is_critical": bool(score >= self.model.threshold),
                "score_is_calibrated": False, "scope": "incident_snapshot_triage", "model_name": self.model.specification["name"],
                "model_sha256": self.metadata["model_sha256"],
                "within_fitting_period": bool(frame.timestamp.iloc[0] <= pd.Timestamp(self.model.fit_through)),
                "outside_fit_ranges": range_evidence(frame.iloc[0], self.model.ranges),
                "limitation": "Incident labels align with source-specific KPI differences; this score is not validated for monitoring alerts or future failures."}
