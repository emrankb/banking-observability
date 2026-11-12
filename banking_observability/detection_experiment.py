"""Train the two detection paths and write evidence from the supplied files."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import pandas as pd

from .anomaly_detection import alert_summary, anomaly_features, fit_anomaly
from .detection_common import (MONITORING_METRICS, deterministic_seed, metric_frame,
                               read_incidents, save_detector, write_json)
from .forecasting import file_sha256, runtime_versions
from .incident_triage import audit_incidents, evaluate_triage, fit_triage


def report_text(summary: dict, triage_metrics: pd.DataFrame) -> str:
    audit, triage, anomaly = summary["incident_audit"], summary["triage"], summary["anomaly"]
    lines = ["# Incident triage and monitoring anomalies", "", "## Incident snapshots", "",
             f"The supplied table contains {audit['incident_rows']} records, including {audit['critical_rows']} marked critical.",
             f"The chronological development interval contains {triage['fit_rows']} records; the later evaluation interval contains {triage['test_rows']}.",
             f"Selected: `{triage['candidate']['name']}` at score threshold {triage['threshold']:.2f}.", "",
             "| Comparison | Records | Critical | Precision | Recall | F1 | Average precision | TN / FP / FN / TP |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in triage_metrics.to_dict("records"):
        ap = "unavailable" if row["average_precision"] is None else f"{row['average_precision']:.4f}"
        lines.append(f"| {row['comparison']} | {row['n']} | {row['critical_count']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {ap} | {row['tn']} / {row['fp']} / {row['fn']} / {row['tp']} |")
    lines += ["", f"There are {audit['disagreeing_rows']} incident records with KPI differences from their exact-timestamp monitoring rows ({audit['disagreeing_cells']} differing cells).",
              f"The exploratory CPU > 85 rule reproduces {audit['cpu_above_85_label_agreements']} of {audit['incident_rows']} supplied labels.",
              "High scenario-classification scores do not establish real incident prediction. The monitoring-snapshot comparison holds the labels fixed while replacing the incident KPIs with the original monitoring readings. Its purpose is to expose the source mismatch, not to evaluate a deployable alert classifier.",
              "The disputed-fields ablation reuses the selected settings without retuning; it is a sensitivity check, not another selected model. Scores have not been probability-calibrated.", "",
              "## Monitoring anomalies", "",
              f"Isolation Forest parameters: `{anomaly['selected_parameters']}`. Parameters were selected for seed-to-seed ranking stability, not detection accuracy.",
              f"The final fit ends at {anomaly['final_fit_through']}; threshold calibration ends at {anomaly['calibrated_through']}.",
              f"The review setting is {anomaly['review_fraction']:.2%} of the calibration score distribution. This is a configurable review-budget assumption, not an estimated incident prevalence.", "",
              "| Interval | Scored minutes | Forest flags | Constant-feature changes | Review minutes | Consecutive episodes | Robust-baseline flags |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("calibration", "test"):
        row = anomaly[name]
        lines.append(f"| {name} | {row['rows']} | {row['forest_flagged_minutes']} | {row['constant_change_minutes']} | {row['review_minutes']} | {row['consecutive_review_episodes']} | {row['robust_baseline_review_minutes']} |")
    lines += ["", "The 60-minute context excludes the current observation from its reference mean. Payment non-success, authentication non-success, application errors, and uptime loss are also scored.",
              "A metric that was constant during fitting cannot be learned as a varying signal by the forest. Changes to such features receive a separate review flag. Reference deviations provide context, not a causal explanation or a decomposition of the forest score.",
              "There are no verified per-minute incident labels for this monitoring series. Precision, recall, F1, and a false-positive rate are therefore not reported for the anomaly detector. A flagged episode means consecutive review minutes, not a confirmed outage.", "",
              "## Reproduction", "",
              "Selection, split boundaries, runtime versions, file hashes, candidate comparisons, and per-row scores are saved beside this report. The existing test interval was previously examined during discovery; it is not an untouched external validation set.",
              "Run `python tests/test_detection.py --data-dir data/raw` to check artifacts, temporal features, threshold calculation, and report counts. The test command trains an independent temporary run rather than changing saved outputs.", ""]
    return "\n".join(lines)


def run_detection(data_dir: Path, model_dir: Path, report_dir: Path, review_fraction: float = 0.01) -> dict:
    data_dir, model_dir, report_dir = (Path(path).resolve() for path in (data_dir, model_dir, report_dir))
    for destination in (model_dir, report_dir):
        if destination == data_dir or destination in data_dir.parents or data_dir in destination.parents:
            raise ValueError("Outputs must be separate from the source-data directory.")
        if destination.exists():
            raise ValueError(f"Output directory already exists: {destination}. Choose a new directory; existing runs are not overwritten.")
    if model_dir == report_dir or model_dir in report_dir.parents or report_dir in model_dir.parents:
        raise ValueError("Model and report directories must be separate, non-nested paths.")
    for name in ("infrastructure_metrics_train.csv", "infrastructure_metrics_test.csv", "incident_reports.csv"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Missing input: {data_dir / name}")
    train_path = data_dir / "infrastructure_metrics_train.csv"
    train = metric_frame(pd.read_csv(train_path), MONITORING_METRICS, step_minutes=1)
    incidents = read_incidents(data_dir / "incident_reports.csv")
    cutoff = train.timestamp.iloc[-1]
    development = incidents.loc[incidents.timestamp.le(cutoff)].reset_index(drop=True)
    evaluation = incidents.loc[incidents.timestamp.gt(cutoff)].reset_index(drop=True)
    if evaluation.empty or development.empty or incidents.timestamp.iloc[0] < train.timestamp.iloc[0]:
        raise ValueError("Incident records must span the monitoring development/evaluation boundary.")
    seed = deterministic_seed(train_path)
    triage, triage_plan, triage_tables = fit_triage(development, seed)
    anomaly, anomaly_plan, anomaly_tables = fit_anomaly(train, seed, review_fraction)
    with tempfile.TemporaryDirectory(prefix="banking-detection-") as scratch:
        scratch = Path(scratch); models = scratch / "models"; reports = scratch / "reports"
        models.mkdir(); reports.mkdir()
        for kind, model, plan, module in (("incident_triage", triage, triage_plan, "incident_triage.py"),
                                           ("monitoring_anomaly", anomaly, anomaly_plan, "anomaly_detection.py")):
            save_detector(models / kind, model, {"kind": kind, "plan": plan,
                          "monitoring_train_sha256": file_sha256(train_path),
                          "incident_file_sha256": file_sha256(data_dir / "incident_reports.csv") if kind == "incident_triage" else None}, module=module)
        # Both decisions and their calibration are fixed before the monitoring test file is read.
        write_json(reports / "selection.json", {"triage": triage_plan, "anomaly": anomaly_plan})
        test = metric_frame(pd.read_csv(data_dir / "infrastructure_metrics_test.csv"), MONITORING_METRICS, step_minutes=1)
        if test.timestamp.iloc[0] != cutoff + pd.Timedelta(minutes=1):
            raise ValueError("Monitoring test data must follow training data by exactly one minute.")
        monitoring = pd.concat([train, test], ignore_index=True)
        audit, differences = audit_incidents(incidents, monitoring)
        metrics, predictions = evaluate_triage(triage, development, evaluation, monitoring, seed)
        all_features = anomaly_features(monitoring)
        evaluation_features = all_features.loc[all_features.index > cutoff]
        test_scores = anomaly.assess(evaluation_features)
        # Keep the recorded monitoring values alongside scores for inspection.
        test_scores = test_scores.merge(test, on="timestamp", how="left", validate="one_to_one")
        tables = {**triage_tables, **anomaly_tables, "incident_discrepancies": differences,
                  "triage_test_metrics": metrics, "triage_test_predictions": predictions, "anomaly_test_scores": test_scores}
        for name, table in tables.items():
            table.to_csv(reports / f"{name}.csv", index=False)
        source_files = ("infrastructure_metrics_train.csv", "infrastructure_metrics_test.csv", "incident_reports.csv")
        summary = {"runtime_versions": runtime_versions(), "source_sha256": {name: file_sha256(data_dir / name) for name in source_files},
                   "incident_audit": audit, "triage": {**triage_plan, "test_rows": len(evaluation), "test_critical": int(evaluation.is_critical.sum())},
                   "anomaly": {**anomaly_plan, "calibration": alert_summary(anomaly_tables["anomaly_calibration"]), "test": alert_summary(test_scores)}}
        write_json(reports / "summary.json", summary)
        (reports / "detection_review.md").write_text(report_text(summary, metrics))
        for destination in (model_dir, report_dir):
            destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(models, model_dir)
        shutil.copytree(reports, report_dir)
    print(f"TRIAGE SELECTED: {triage.specification['name']}; threshold={triage.threshold:.2f}")
    print(metrics.to_string(index=False))
    print("\nMONITORING ANOMALY REVIEW:", summary["anomaly"]["test"])
    print("Incident triage is scenario-only; monitoring review flags have no verified event labels.")
    print("Report:", report_dir / "detection_review.md")
    return summary
