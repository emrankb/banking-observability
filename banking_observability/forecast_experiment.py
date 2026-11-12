"""Training-only temporal model selection, followed by a frozen replay evaluation."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from .forecast_features import (
    FeatureSpec, REQUIRED_COLUMNS, TARGET, build_features, make_supervised, read_history,
)
from .forecast_models import candidate_specs, fit_candidate
from .forecasting import LatencyForecaster, file_sha256, runtime_versions


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) != len(predicted) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Scoring requires equally sized finite actuals and predictions.")
    return {
        "n": int(len(actual)), "mae_ms": float(mean_absolute_error(actual, predicted)),
        "rmse_ms": float(root_mean_squared_error(actual, predicted)),
        "bias_ms": float(np.mean(predicted - actual)),
        "p95_absolute_error_ms": float(np.quantile(np.abs(actual - predicted), 0.95)),
        "r2": float(r2_score(actual, predicted)),
    }


def temporal_folds(features: pd.DataFrame, *, horizon: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = TimeSeriesSplit(n_splits=3, test_size=1440, gap=horizon)
    folds = list(splitter.split(features))
    for train, validation in folds:
        last_training_label = features.index[train[-1]] + pd.Timedelta(minutes=horizon)
        if last_training_label >= features.index[validation[0]]:
            raise AssertionError("Training target reaches a validation forecast origin.")
    return folds


def training_diagnostics(train: pd.DataFrame, output: Path) -> dict:
    latency = train[TARGET]
    # Diagnostic only: these training-wide means are NOT used as model features.
    bucket = train.timestamp.dt.hour * 4 + train.timestamp.dt.minute // 15
    conditional_residual = latency - latency.groupby(bucket).transform("mean")
    diagnostic = {
        "target_column": TARGET, "train_rows": len(train),
        "train_start": train.timestamp.iloc[0].isoformat(), "train_end": train.timestamp.iloc[-1].isoformat(),
        "autocorrelation": {str(lag): float(latency.autocorr(lag)) for lag in (1, 5, 15, 60, 1440)},
        "training_clock_demeaned_autocorrelation": {str(lag): float(conditional_residual.autocorr(lag)) for lag in (1, 5, 15, 60, 1440)},
        "cpu_same_minute_pearson": float(train.cpu_utilization.corr(latency)),
        "cpu_vs_five_minute_future_pearson": float(train.cpu_utilization.corr(latency.shift(-5))),
        "floor_100ms_rows": int(latency.eq(100).sum()),
        "assumption": "Conditional patterns may repeat; neither global nor future stationarity is established by this short history.",
    }
    write_json(output / "training_diagnostics.json", diagnostic)
    days = train.assign(day=train.timestamp.dt.strftime("%Y-%m-%d")).groupby("day")[TARGET].agg(["count", "mean", "std", "min", "max"])
    days.to_csv(output / "training_daily_summary.csv")
    hours = train.assign(hour=train.timestamp.dt.hour, weekend=train.timestamp.dt.dayofweek.ge(5)).groupby(["weekend", "hour"])[TARGET].agg(["count", "mean", "std"])
    hours.to_csv(output / "training_hourly_summary.csv")
    return diagnostic


def train_and_evaluate(data_dir: Path, output: Path, models: Path, *, overwrite: bool = False) -> dict:
    if not overwrite and ((output / "selection.json").exists() or (models / "metadata.json").exists()):
        raise ValueError("A checkpoint already exists. Use new output/model directories or explicitly pass --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    spec = FeatureSpec()
    train_path = data_dir / "infrastructure_metrics_train.csv"
    train = read_history(train_path, minimum_rows=spec.minimum_history_rows)
    train_hash = file_sha256(train_path)
    seed = int(train_hash[:8], 16)
    features, target = make_supervised(train, spec)
    folds = temporal_folds(features, horizon=spec.horizon_minutes)
    candidates = candidate_specs()
    plan = {
        "target_column": TARGET, "horizon_minutes": spec.horizon_minutes,
        "target_definition": "single minute observation at origin + 5 minutes, not a five-minute average",
        "availability_assumption": "All t readings have arrived at t; no undocumented ingestion delay is simulated.",
        "source_clock": "timezone-naive, timezone unspecified",
        "feature_spec": spec.to_dict(), "minimum_history_rows": spec.minimum_history_rows,
        "forecast_origin_uses_latest_reading": True,
        "rolling_windows": "trailing, inclusive of origin; never centered",
        "selection": "lowest mean validation MAE, then mean RMSE, then candidate name; all folds are equal length",
        "n_temporal_folds": 3, "validation_rows_per_fold": 1440, "purged_origin_rows": spec.horizon_minutes,
        "candidates": candidates,
        "seed": seed, "seed_rationale": "First 8 hexadecimal digits of training-file SHA256; fixed reproducibility seed, not tuned.",
        "training_source_sha256": train_hash,
        "evaluation": "Fixed fitted model; one new five-minute forecast per observed minute. Earlier test readings may become history; no test refitting.",
        "test_eligibility": "Forecast origin >= first supplied test timestamp, and target <= last supplied test timestamp.",
        "test_selection_policy": "Freeze winner and group-best comparators before reading the test file in this script.",
        "test_inspection_disclosure": "Phase 1 already explored the full monitoring interval. This is a reserved model-evaluation split, not a discovery-blind holdout.",
        "interval_policy": "Point forecasts only; no calibrated uncertainty interval is claimed.",
        "post_test_tuning": "None in this checkpoint; test metrics cannot change the selected artifact.",
        "runtime_versions": runtime_versions(), "python": platform.python_version(),
    }
    # Persist the full experiment definition BEFORE fitting or evaluating candidates.
    write_json(output / "experiment_plan.json", plan)
    diagnostics = training_diagnostics(train, output)
    fold_bounds, cv_rows, cv_predictions = [], [], []
    for number, (fit_rows, valid_rows) in enumerate(folds, 1):
        fold_bounds.append({
            "fold": number, "train_rows": len(fit_rows), "validation_rows": len(valid_rows),
            "train_origin_start": features.index[fit_rows[0]].isoformat(),
            "train_origin_end": features.index[fit_rows[-1]].isoformat(),
            "train_target_end": (features.index[fit_rows[-1]] + pd.Timedelta(minutes=5)).isoformat(),
            "validation_origin_start": features.index[valid_rows[0]].isoformat(),
            "validation_origin_end": features.index[valid_rows[-1]].isoformat(),
            "validation_target_end": (features.index[valid_rows[-1]] + pd.Timedelta(minutes=5)).isoformat(),
            "purged_origin_rows": spec.horizon_minutes,
        })
    pd.DataFrame(fold_bounds).to_csv(output / "folds.csv", index=False)
    with threadpool_limits(limits=1):
        for candidate in candidates:
            fold_metrics = []
            for number, (fit_rows, valid_rows) in enumerate(folds, 1):
                fitted = fit_candidate(candidate, features.iloc[fit_rows], target.iloc[fit_rows], seed=seed)
                predicted = fitted.predict(features.iloc[valid_rows])
                metrics = score(target.iloc[valid_rows], predicted)
                cv_rows.append({"candidate": candidate["name"], "kind": candidate["kind"], "group": candidate["group"], "fold": number, **metrics})
                fold_metrics.append(metrics["mae_ms"])
                cv_predictions.append(pd.DataFrame({
                    "candidate": candidate["name"], "fold": number,
                    "forecast_origin": features.index[valid_rows],
                    "target_timestamp": features.index[valid_rows] + pd.Timedelta(minutes=5),
                    "actual_ms": target.iloc[valid_rows].to_numpy(), "prediction_ms": predicted,
                }))
            print(f"CV {candidate['name']}: MAE {np.mean(fold_metrics):.4f} ms; folds " + ", ".join(f"{x:.4f}" for x in fold_metrics), flush=True)
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(output / "cv_fold_metrics.csv", index=False)
    pd.concat(cv_predictions, ignore_index=True).to_csv(output / "cv_predictions.csv", index=False)
    leaderboard = cv_table.groupby(["candidate", "kind", "group"], as_index=False).agg(
        mean_mae_ms=("mae_ms", "mean"), mean_rmse_ms=("rmse_ms", "mean"),
        min_fold_mae_ms=("mae_ms", "min"), max_fold_mae_ms=("mae_ms", "max"),
    ).sort_values(["mean_mae_ms", "mean_rmse_ms", "candidate"], kind="stable").reset_index(drop=True)
    leaderboard.to_csv(output / "cv_leaderboard.csv", index=False)
    winner_name = str(leaderboard.iloc[0]["candidate"])
    by_name = {candidate["name"]: candidate for candidate in candidates}
    winner_spec = by_name[winner_name]
    comparator_names = [c["name"] for c in candidates if c["kind"] not in ("hgb", "ridge")]
    group_best = {}
    for group in ("clock", "latency_clock", "all"):
        group_best[group] = str(leaderboard.loc[leaderboard.kind.eq("hgb") & leaderboard.group.eq(group)].iloc[0]["candidate"])
        comparator_names.append(group_best[group])
    best_ridge = str(leaderboard.loc[leaderboard.kind.eq("ridge")].iloc[0]["candidate"])
    comparator_names.extend([best_ridge, winner_name])
    comparator_names = list(dict.fromkeys(comparator_names))
    selection = {
        "selected_candidate": winner_spec,
        "selected_cv_mean_mae_ms": float(leaderboard.iloc[0]["mean_mae_ms"]),
        "group_best_hgb_candidates": group_best, "test_comparators": comparator_names,
        "selection_basis": "training-only purged temporal validation; test not yet read by this script",
        "fit_origin_start": features.index[0].isoformat(), "fit_origin_end": features.index[-1].isoformat(),
        "fit_target_end": (features.index[-1] + pd.Timedelta(minutes=5)).isoformat(),
        "fit_supervised_rows": len(features), "raw_training_rows": len(train),
        "warmup_rows": spec.minimum_history_rows - 1,
        "unavailable_future_targets_at_training_end": spec.horizon_minutes,
    }
    write_json(output / "selection.json", selection)
    with threadpool_limits(limits=1):
        final_models = {name: fit_candidate(by_name[name], features, target, seed=seed) for name in comparator_names}
    model_path = models / "model.joblib"
    temporary_path = models / "model.joblib.tmp"
    joblib.dump(final_models[winner_name], temporary_path, compress=3)
    temporary_path.replace(model_path)
    metadata = {
        "format_version": 1, "candidate": winner_spec,
        "target_column": TARGET, "feature_spec": spec.to_dict(),
        "all_feature_columns": features.columns.tolist(),
        "selected_feature_columns": final_models[winner_name].columns,
        "minimum_history_rows": spec.minimum_history_rows,
        "runtime_versions": runtime_versions(), "python": platform.python_version(),
        "model_sha256": file_sha256(model_path),
        "feature_code_sha256": file_sha256(Path(__file__).with_name("forecast_features.py")),
        "training_source_sha256": train_hash,
        "trained_target_through": selection["fit_target_end"],
        "training_metric_ranges": {c: {"min": float(train[c].min()), "max": float(train[c].max())} for c in REQUIRED_COLUMNS[1:]},
        "evaluation_contract": plan["evaluation"],
        "trusted_artifacts_only": "joblib uses pickle; checksum verification is not a security sandbox.",
        "selection_cv_mean_mae_ms": selection["selected_cv_mean_mae_ms"],
    }
    metadata["role"] = "validation-selected serving candidate"
    write_json(models / "metadata.json", metadata)
    registry = [{"name": winner_name, "path": ".", "role": metadata["role"]}]
    for name in comparator_names:
        if name == winner_name or by_name[name]["kind"] not in {"ridge", "hgb"}:
            continue
        candidate_dir = models / "comparators" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_models[name], candidate_dir / "model.joblib", compress=3)
        candidate_metadata = dict(metadata)
        candidate_metadata.update({
            "candidate": by_name[name], "selected_feature_columns": final_models[name].columns,
            "model_sha256": file_sha256(candidate_dir / "model.joblib"),
            "selection_cv_mean_mae_ms": float(leaderboard.loc[leaderboard.candidate.eq(name), "mean_mae_ms"].iloc[0]),
            "role": "trained comparator; not selected for serving",
        })
        write_json(candidate_dir / "metadata.json", candidate_metadata)
        registry.append({"name": name, "path": str(Path("comparators") / name), "role": candidate_metadata["role"]})
    write_json(models / "registry.json", registry)
    # Model selection is complete. Only now load the supplied model-evaluation split.
    test_path = data_dir / "infrastructure_metrics_test.csv"
    test = read_history(test_path)
    if test.timestamp.iloc[0] != train.timestamp.iloc[-1] + pd.Timedelta(minutes=1):
        raise ValueError("Train and test must be consecutive, non-overlapping minute histories.")
    combined = pd.concat([train, test], ignore_index=True)
    all_features, all_targets = make_supervised(combined, spec)
    test_features = all_features.loc[all_features.index >= test.timestamp.iloc[0]]
    test_targets = all_targets.loc[test_features.index]
    evaluation = []
    prediction_table = pd.DataFrame({
        "forecast_origin": test_features.index,
        "target_timestamp": test_features.index + pd.Timedelta(minutes=5),
        "actual_ms": test_targets.to_numpy(),
    })
    with threadpool_limits(limits=1):
        for name in comparator_names:
            predicted = final_models[name].predict(test_features)
            prediction_table[name] = predicted
            evaluation.append({"candidate": name, "selected": name == winner_name, **score(test_targets, predicted)})
    prediction_table.to_csv(output / "test_predictions.csv", index=False)
    evaluation_table = pd.DataFrame(evaluation).sort_values("mae_ms")
    evaluation_table.to_csv(output / "test_metrics.csv", index=False)
    # Descriptive slices do not alter model selection or define operational SLOs.
    target_time = pd.DatetimeIndex(prediction_table.target_timestamp)
    slice_masks = {"all": np.ones(len(target_time), dtype=bool)}
    for day in sorted(set(target_time.strftime("%Y-%m-%d"))):
        slice_masks[f"day:{day}"] = target_time.strftime("%Y-%m-%d") == day
    slice_masks["recorded_hours:09_to_17"] = (target_time.hour >= 9) & (target_time.hour < 18)
    slice_masks["recorded_hours:other"] = ~slice_masks["recorded_hours:09_to_17"]
    p90 = float(train[TARGET].quantile(0.90))
    slice_masks["actual_above_training_p90"] = test_targets.to_numpy() > p90
    slice_masks["near_09_or_18_boundary_5min"] = np.logical_or.reduce([
        np.abs(target_time.hour * 60 + target_time.minute - minute) <= 5 for minute in (540, 1080)
    ])
    slice_rows = []
    for label, mask in slice_masks.items():
        if int(mask.sum()) < 2:
            continue
        for name in comparator_names:
            slice_rows.append({"slice": label, "candidate": name, **score(test_targets.to_numpy()[mask], prediction_table[name].to_numpy()[mask])})
    pd.DataFrame(slice_rows).to_csv(output / "test_slices.csv", index=False)
    variation = []
    buckets = target_time.floor("15min")
    for name in ["actual_ms", *comparator_names]:
        series = pd.Series(prediction_table[name].to_numpy(), index=test_features.index)
        within = series - series.groupby(buckets).transform("mean")
        variation.append({"series": name, "std_ms": float(series.std(ddof=0)), "first_difference_std_ms": float(series.diff().dropna().std(ddof=0)), "within_15min_bucket_std_ms": float(within.std(ddof=0))})
    pd.DataFrame(variation).to_csv(output / "forecast_variation.csv", index=False)
    loaded = LatencyForecaster(models)
    with threadpool_limits(limits=1):
        reload_prediction = loaded.model.predict(test_features)
    np.testing.assert_allclose(reload_prediction, prediction_table[winner_name], rtol=0, atol=1e-10)
    example_origin = test.timestamp.iloc[0]
    example = loaded.predict(combined, as_of=example_origin)
    write_json(output / "inference_example.json", example)
    # Exercise the common inference loader for every persisted trained comparator.
    for entry in registry:
        comparator = LatencyForecaster(models / entry["path"])
        with threadpool_limits(limits=1):
            np.testing.assert_allclose(
                comparator.model.predict(test_features), prediction_table[entry["name"]],
                rtol=0, atol=1e-10,
            )
    selected_score = next(row for row in evaluation if row["selected"])
    persistence_score = next(row for row in evaluation if row["candidate"] == "persistence")
    seasonal_score = next(row for row in evaluation if row["candidate"] == "daily_seasonal_naive")
    summary = {
        "selected_candidate": winner_name, "selected_cv_mean_mae_ms": selection["selected_cv_mean_mae_ms"],
        "selected_test_metrics": selected_score,
        "test_origin_start": test_features.index[0].isoformat(), "test_origin_end": test_features.index[-1].isoformat(),
        "test_target_start": prediction_table.target_timestamp.iloc[0].isoformat(),
        "test_target_end": prediction_table.target_timestamp.iloc[-1].isoformat(),
        "raw_test_rows": len(test), "evaluated_forecasts": len(test_features),
        "unscored_final_origins_without_target": spec.horizon_minutes,
        "test_candidate_count": len(evaluation), "cv_candidate_count": len(candidates),
        "temporal_fits": len(candidates) * len(folds),
        "mae_improvement_vs_persistence_pct": 100 * (1 - selected_score["mae_ms"] / persistence_score["mae_ms"]),
        "mae_improvement_vs_daily_seasonal_pct": 100 * (1 - selected_score["mae_ms"] / seasonal_score["mae_ms"]),
        "training_p90_ms": p90,
        "source_sha256": {train_path.name: train_hash, test_path.name: file_sha256(test_path)},
        "test_replay_is_not_fixed_origin_multistep": True,
        "reload_matches_in_memory_atol": 1e-10,
        "diagnostics": diagnostics,
    }
    write_json(output / "summary.json", summary)
    print("\nFROZEN SELECTED MODEL:", winner_name, flush=True)
    print(evaluation_table.to_string(index=False), flush=True)
    return summary
