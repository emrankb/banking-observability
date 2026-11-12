"""Explicit baseline and estimator definitions; every candidate is evaluated."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .forecast_features import feature_columns


@dataclass
class FittedCandidate:
    specification: dict
    columns: list[str]
    estimator: object | None = None
    profile: dict[int, float] | None = None
    constant: float | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        kind = self.specification["kind"]
        if kind == "global_mean":
            result = np.full(len(features), self.constant, dtype=float)
        elif kind == "persistence":
            result = features["latency__lag_0"].to_numpy()
        elif kind == "rolling_mean":
            result = features["latency__mean_15"].to_numpy()
        elif kind == "seasonal_naive":
            result = features["latency__previous_day_at_target"].to_numpy()
        elif kind == "clock_profile":
            result = features["clock__quarter_hour"].map(self.profile).fillna(self.constant).to_numpy()
        else:
            if self.estimator is None:
                raise RuntimeError("Fitted estimator is missing.")
            result = self.estimator.predict(features.loc[:, self.columns])
        result = np.asarray(result, dtype=float)
        if not np.isfinite(result).all():
            raise RuntimeError("Model returned non-finite predictions.")
        # No clipping, rounding, artificial jitter, or use of later observations.
        return result


def candidate_specs() -> list[dict]:
    specs = [
        {"name": "global_mean", "kind": "global_mean", "group": "all", "params": {}},
        {"name": "persistence", "kind": "persistence", "group": "all", "params": {}},
        {"name": "rolling_mean_15", "kind": "rolling_mean", "group": "all", "params": {}},
        {"name": "daily_seasonal_naive", "kind": "seasonal_naive", "group": "all", "params": {}},
        {"name": "clock_profile_15", "kind": "clock_profile", "group": "clock", "params": {}},
    ]
    for alpha in (0.1, 10.0, 1000.0):
        specs.append({"name": f"ridge_all_alpha_{alpha:g}", "kind": "ridge", "group": "all", "params": {"alpha": alpha}})
    # Small, predeclared search: two tree sizes and two regularization budgets.
    # The paired budgets are candidates, not a claim to isolate individual knobs.
    for group in ("clock", "latency_clock", "all"):
        for leaves in (7, 15):
            for budget, leaf_rows, iterations, regularization in (
                ("compact", 60, 150, 1.0), ("regularized", 120, 300, 10.0),
            ):
                specs.append({
                    "name": f"hgb_{group}_leaves_{leaves}_{budget}", "kind": "hgb", "group": group,
                    "params": {
                        "loss": "absolute_error", "learning_rate": 0.05,
                        "max_iter": iterations, "max_leaf_nodes": leaves,
                        "min_samples_leaf": leaf_rows, "l2_regularization": regularization,
                        "max_bins": 128, "early_stopping": False,
                    },
                })
    return specs


def fit_candidate(spec: dict, features: pd.DataFrame, target: pd.Series, *, seed: int) -> FittedCandidate:
    if not features.index.equals(target.index):
        raise ValueError("Training feature and target origins are misaligned.")
    if len(features) == 0 or not np.isfinite(features.to_numpy()).all() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("Training requires nonempty finite features and targets.")
    baseline_columns = {
        "global_mean": [], "persistence": ["latency__lag_0"],
        "rolling_mean": ["latency__mean_15"],
        "seasonal_naive": ["latency__previous_day_at_target"],
        "clock_profile": ["clock__quarter_hour"],
    }
    columns = baseline_columns.get(spec["kind"], feature_columns(features, spec["group"]))
    fitted = FittedCandidate(specification=spec, columns=columns)
    if spec["kind"] == "global_mean":
        fitted.constant = float(target.mean())
    elif spec["kind"] == "clock_profile":
        profile = target.groupby(features["clock__quarter_hour"]).median()
        fitted.profile = {int(k): float(v) for k, v in profile.items()}
        fitted.constant = float(target.median())
    elif spec["kind"] == "ridge":
        fitted.estimator = make_pipeline(StandardScaler(), Ridge(solver="svd", **spec["params"]))
    elif spec["kind"] == "hgb":
        fitted.estimator = HistGradientBoostingRegressor(random_state=seed, **spec["params"])
    elif spec["kind"] not in {"persistence", "rolling_mean", "seasonal_naive"}:
        raise ValueError("Unsupported candidate kind: " + spec["kind"])
    if fitted.estimator is not None:
        fitted.estimator.fit(features.loc[:, columns], target)
    return fitted
