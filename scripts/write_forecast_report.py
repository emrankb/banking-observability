#!/usr/bin/env python3
"""Summarize a saved forecast experiment and its source-checked predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.forecast_experiment import score  # noqa: E402
from banking_observability.forecast_features import TARGET, read_history  # noqa: E402
from banking_observability.forecasting import LatencyForecaster, file_sha256  # noqa: E402


def table(frame: pd.DataFrame) -> str:
    def display(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(display(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def write_report(report_dir: Path, model_dir: Path, data_dir: Path) -> Path:
    summary = json.loads((report_dir / "summary.json").read_text())
    selection = json.loads((report_dir / "selection.json").read_text())
    plan = json.loads((report_dir / "experiment_plan.json").read_text())
    model = LatencyForecaster(model_dir)
    cv = pd.read_csv(report_dir / "cv_leaderboard.csv")
    folds = pd.read_csv(report_dir / "folds.csv")
    metrics = pd.read_csv(report_dir / "test_metrics.csv")
    predictions = pd.read_csv(report_dir / "test_predictions.csv", parse_dates=["target_timestamp"])
    registry = json.loads((model_dir / "registry.json").read_text())
    for name, expected in summary["source_sha256"].items():
        if file_sha256(data_dir / name) != expected:
            raise ValueError(f"Source fingerprint mismatch: {name}. Recreate the experiment before reporting.")
    if model.metadata["training_source_sha256"] != summary["source_sha256"]["infrastructure_metrics_train.csv"]:
        raise ValueError("Model and report use different training sources.")
    candidate = selection["selected_candidate"]["name"]
    if candidate != summary["selected_candidate"] or candidate != model.metadata["candidate"]["name"]:
        raise ValueError("Selected candidate differs between the report and artifact.")
    train = read_history(data_dir / "infrastructure_metrics_train.csv")
    test = read_history(data_dir / "infrastructure_metrics_test.csv")
    history = pd.concat([train, test], ignore_index=True).set_index("timestamp")
    actual = history[TARGET].reindex(predictions.target_timestamp).to_numpy()
    if not np.isfinite(actual).all() or not np.array_equal(actual, predictions.actual_ms.to_numpy()):
        raise ValueError("Saved actuals do not match the source timestamps.")
    for row in metrics.to_dict("records"):
        recomputed = score(actual, predictions[row["candidate"]])
        if any(not np.isclose(float(row[key]), value, rtol=0, atol=1e-9)
               for key, value in recomputed.items()):
            raise ValueError(f"Saved evaluation metrics do not match predictions for {row['candidate']}.")
    chosen = metrics.loc[metrics.candidate.eq(candidate)].iloc[0]
    board = cv.sort_values(["mean_mae_ms", "mean_rmse_ms", "candidate"])
    if board.iloc[0].candidate != candidate:
        raise ValueError("Artifact is not the validation-selected candidate.")
    role = "learned estimator" if selection["selected_candidate"]["kind"] in {"ridge", "hgb"} else "baseline"
    same_hours = train.loc[train.timestamp.dt.hour.between(9, 13)].copy()
    same_hours["calendar_group"] = same_hours.timestamp.dt.dayofweek.ge(5).map({False: "weekday", True: "weekend"})
    regime = same_hours.groupby("calendar_group").agg(
        readings=(TARGET, "size"), mean_processing_ms=(TARGET, "mean"), mean_cpu=("cpu_utilization", "mean")
    ).reset_index()
    column_list = ", ".join(f"`{name}`" for name in model.model.columns) or "none (constant baseline)"
    text = f'''# Five-minute processing-time forecast

## Selected candidate

`{candidate}` ({role}) was selected by mean chronological-validation MAE:
**{selection['selected_cv_mean_mae_ms']:.4f} ms**. On {len(predictions):,} reserved
evaluation origins, MAE is **{chosen.mae_ms:.4f} ms** and RMSE is **{chosen.rmse_ms:.4f} ms**.
These values are recomputed from saved predictions and checked against source readings.

The target is the single `{TARGET}` reading five minutes after the origin, not
a future-window average or endpoint latency. Observations through the origin
are assumed available. Timezone and ingestion delays are unspecified.

## Validation

{table(folds[['fold', 'train_rows', 'validation_rows', 'train_target_end', 'validation_origin_start', 'validation_target_end']])}

The experiment compared {summary['cv_candidate_count']} candidates with
{summary['temporal_fits']} temporal fits. Five rows between fitting and validation
keep training labels before the next validation origin. All candidates share
{len(model.metadata['all_feature_columns'])} engineered feature columns and the same
eligible origins; feature subsets differ by candidate. The serving candidate uses: {column_list}.

{table(board[['candidate', 'mean_mae_ms', 'mean_rmse_ms', 'max_fold_mae_ms']])}

## Training regimes

For the same recorded hours, 09:00–13:59:

{table(regime)}

Training five-minute target autocorrelation is
{summary['diagnostics']['autocorrelation']['5']:.4f}. Short-history autocorrelation
does not establish a stable weekly pattern. Review the per-fold errors rather
than assuming a weekday relationship will transfer to the weekend.

## Reserved evaluation

{table(metrics[['candidate', 'selected', 'n', 'mae_ms', 'rmse_ms', 'bias_ms', 'p95_absolute_error_ms']])}

The serving choice remains the validation winner even if the test ranking differs.
Each origin receives a new five-minute forecast using readings already observed;
no estimator is refit during replay. Discovery previously inspected this interval,
so it is not a discovery-blind holdout. The empirical 95th-percentile absolute error
is an error summary, not a calibrated prediction interval.

## Saved artifacts

{table(pd.DataFrame(registry))}

`metadata.json` records source and artifact fingerprints, feature order, runtime
versions, the feature-code fingerprint, and the training-target cutoff. The shared
inference interface requires {model.spec.minimum_history_rows:,} consecutive minute
readings for daily-baseline and learned-candidate compatibility. A trailing-average
baseline itself needs fewer readings. Only load trusted joblib artifacts; a matching
checksum does not make pickle deserialization safe against a malicious publisher.

## Reproduction

The experiment definition is in `experiment_plan.json`. Validation predictions,
evaluation predictions, and slice errors are saved beside this report. Running
`scripts/write_forecast_report.py` checks the supplied source fingerprints and
recalculates evaluation scores before writing this summary.

This report does not assert that the tests or notebook have run. Execute
`tests/test_forecasting.py` and the notebook separately to verify them in the
current environment. No live telemetry or API is involved in this experiment.
'''
    path = report_dir / "forecast_review.md"
    path.write_text(text, encoding="utf-8")
    card = f'''# {candidate}

Role: validation-selected {role}.
Target: `{TARGET}` at origin + {model.spec.horizon_minutes} minutes.
Actual predictive features: {column_list}.

Validation MAE: {selection['selected_cv_mean_mae_ms']:.4f} ms.
Reserved evaluation: {len(predictions):,} origins, MAE {chosen.mae_ms:.4f} ms,
RMSE {chosen.rmse_ms:.4f} ms. Selection did not use test rank.

The common interface requires {model.spec.minimum_history_rows:,} complete monitoring
readings. The training-target cutoff is {model.metadata['trained_target_through']}.
For a nonlearned baseline this is the experiment's common selection cutoff, not
a claim that baseline parameters were learned.

Point forecasts only. Limited weekly coverage, unspecified timezone and ingestion
delay, and no independent later-week evaluation. This is not an incident predictor.
See `metadata.json` and `registry.json`. Load only trusted model files.
'''
    (model_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/forecast")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/latency_5min")
    args = parser.parse_args()
    try:
        print(write_report(args.report_dir, args.model_dir, args.data_dir))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"Forecast report failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
