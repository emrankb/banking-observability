#!/usr/bin/env python3
"""Exercise both detection models with the supplied data and fresh temporary runs."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.anomaly_detection import (
    CANDIDATES as ANOMALY_CANDIDATES, LOAD_METRICS, MonitoringDetector,
    alert_summary, anomaly_features, forest, rank_stability, review_threshold,
)
from banking_observability.detection_common import (
    DISPUTED_METRICS, INCIDENT_METRICS, MONITORING_METRICS, deterministic_seed,
    metric_frame, read_incidents, write_json,
)
from banking_observability.forecasting import file_sha256
from banking_observability.incident_triage import (
    CANDIDATES, IncidentTriage, audit_incidents, classification_metrics, estimator,
)

DATA_DIR: Path


def cli(script: str, *args: object, timeout=120):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_hashes = {p.name: file_sha256(p) for p in DATA_DIR.glob("*.csv")}
        cls.work = tempfile.TemporaryDirectory(prefix="detection-tests-")
        cls.addClassCleanup(cls.work.cleanup)
        cls.directory = Path(cls.work.name)
        cls.models = cls.directory / "models"; cls.reports = cls.directory / "reports"
        run = cli("train_detection.py", "--data-dir", DATA_DIR, "--model-dir", cls.models, "--report-dir", cls.reports)
        if run.returncode or run.stderr:
            raise RuntimeError(f"Training failed or warned:\n{run.stdout}\n{run.stderr}")
        cls.train_stdout = run.stdout
        cls.train = metric_frame(pd.read_csv(DATA_DIR / "infrastructure_metrics_train.csv"), MONITORING_METRICS, step_minutes=1)
        cls.test = metric_frame(pd.read_csv(DATA_DIR / "infrastructure_metrics_test.csv"), MONITORING_METRICS, step_minutes=1)
        cls.history = pd.concat([cls.train, cls.test], ignore_index=True)
        cls.incidents = read_incidents(DATA_DIR / "incident_reports.csv")
        cls.cutoff = cls.train.timestamp.iloc[-1]
        cls.development = cls.incidents.loc[cls.incidents.timestamp.le(cls.cutoff)].reset_index(drop=True)
        cls.evaluation = cls.incidents.loc[cls.incidents.timestamp.gt(cls.cutoff)].reset_index(drop=True)
        cls.summary = json.loads((cls.reports / "summary.json").read_text())
        cls.selection = json.loads((cls.reports / "selection.json").read_text())
        cls.triage = IncidentTriage(cls.models / "incident_triage")
        cls.anomaly = MonitoringDetector(cls.models / "monitoring_anomaly")
        cls.features = anomaly_features(cls.history)

    def test_training_and_cli_success_output(self):
        self.assertIn("TRIAGE SELECTED:", self.train_stdout)
        self.assertIn("MONITORING ANOMALY REVIEW:", self.train_stdout)

    def test_source_hashes_are_unchanged(self):
        self.assertEqual(self.original_hashes, {p.name: file_sha256(p) for p in DATA_DIR.glob("*.csv")})
        for name, sha in self.summary["source_sha256"].items():
            self.assertEqual(sha, self.original_hashes[name])

    def test_incident_target_is_supplied_not_rebuilt(self):
        raw = pd.read_csv(DATA_DIR / "incident_reports.csv")
        np.testing.assert_array_equal(raw.is_critical, self.incidents.is_critical)
        self.assertEqual(self.triage.model.columns, list(INCIDENT_METRICS))
        self.assertFalse({"incident_id", "severity", "incident_type", "is_critical", "timestamp"} & set(self.triage.model.columns))

    def test_incident_split_uses_monitoring_boundary(self):
        self.assertLessEqual(self.development.timestamp.max(), self.cutoff)
        self.assertGreater(self.evaluation.timestamp.min(), self.cutoff)
        self.assertEqual(len(self.development) + len(self.evaluation), len(self.incidents))

    def test_triage_folds_and_class_weights(self):
        saved = pd.read_csv(self.reports / "triage_folds.csv")
        for i, (a, b) in enumerate(TimeSeriesSplit(n_splits=3, test_size=24, gap=1).split(self.development)):
            fit, validation = self.development.iloc[a], self.development.iloc[b]
            self.assertLess(fit.timestamp.max(), validation.timestamp.min())
            self.assertEqual(b[0] - a[-1] - 1, 1)
            self.assertAlmostEqual(saved.iloc[i].critical_weight, len(fit) / (2 * int(fit.is_critical.sum())))
            self.assertAlmostEqual(saved.iloc[i].noncritical_weight, len(fit) / (2 * int((1-fit.is_critical).sum())))
        for spec in CANDIDATES:
            model = estimator(spec, deterministic_seed(DATA_DIR / "infrastructure_metrics_train.csv"))
            fitted = model[-1] if spec["kind"] == "logistic" else model
            self.assertEqual(fitted.class_weight, "balanced")

    def test_triage_metrics_and_threshold_selection(self):
        predictions = pd.read_csv(self.reports / "triage_cv_predictions.csv")
        metrics = pd.read_csv(self.reports / "triage_cv_metrics.csv")
        for row in metrics.to_dict("records"):
            group = predictions.loc[predictions.candidate.eq(row["candidate"]) & predictions.fold.eq(row["fold"])]
            actual = self.development.set_index("timestamp").is_critical.reindex(pd.to_datetime(group.timestamp)).to_numpy()
            np.testing.assert_array_equal(actual, group.actual)
            values = classification_metrics(actual, group.critical_score, row["threshold"])
            for key, value in values.items():
                if value is not None:
                    self.assertAlmostEqual(row[key], value, places=10)
        board = pd.read_csv(self.reports / "triage_cv_leaderboard.csv")
        winner = board.sort_values(["f2", "average_precision", "precision", "order", "distance_from_half", "threshold"],
                                  ascending=[False, False, False, True, True, True]).iloc[0]
        self.assertEqual(self.triage.model.specification["name"], winner.candidate)
        self.assertEqual(self.triage.model.threshold, winner.threshold)

    def test_scaler_sees_only_development_records(self):
        if self.triage.model.specification["kind"] == "logistic":
            scaler = self.triage.model.estimator[0]
            self.assertEqual(scaler.n_samples_seen_, len(self.development))
            np.testing.assert_allclose(scaler.mean_, self.development[list(INCIDENT_METRICS)].mean(), atol=1e-10)

    def test_incident_audit_recomputes_actual_discrepancies(self):
        audit, cells = audit_incidents(self.incidents, self.history)
        self.assertEqual(audit, self.summary["incident_audit"])
        self.assertEqual(cells.is_critical.unique().tolist(), [1])
        self.assertEqual(set(cells.metric), set(DISPUTED_METRICS))
        self.assertEqual(audit["disagreeing_rows"], int(self.incidents.is_critical.sum()))
        self.assertEqual(audit["cpu_above_85_label_agreements"], len(self.incidents))

    def test_saved_triage_predictions_recompute(self):
        predictions = pd.read_csv(self.reports / "triage_test_predictions.csv")
        group = predictions.loc[predictions.comparison.eq("selected_incident_model")]
        np.testing.assert_allclose(group.critical_score, self.triage.model.score(self.evaluation), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(group.actual, self.evaluation.is_critical)
        metrics = pd.read_csv(self.reports / "triage_test_metrics.csv")
        for row in metrics.to_dict("records"):
            group = predictions.loc[predictions.comparison.eq(row["comparison"])]
            recomputed = classification_metrics(group.actual, group.critical_score, row["threshold"])
            for name, value in recomputed.items():
                if value is not None:
                    self.assertAlmostEqual(row[name], value, places=10)

    def test_monitoring_replacement_is_same_timestamp_not_altered_data(self):
        scores = pd.read_csv(self.reports / "triage_test_predictions.csv")
        group = scores.loc[scores.comparison.eq("same_model_monitoring_snapshots")]
        monitoring = self.evaluation[["timestamp"]].merge(self.history, on="timestamp", validate="one_to_one")
        np.testing.assert_allclose(group.critical_score, self.triage.model.score(monitoring), atol=1e-12, rtol=0)

    def test_ablation_excludes_disputed_fields(self):
        cols = [c for c in INCIDENT_METRICS if c not in DISPUTED_METRICS]
        model = estimator(self.triage.model.specification, self.selection["triage"]["seed"])
        with threadpool_limits(limits=1):
            model.fit(self.development[cols], self.development.is_critical)
            expected = model.predict_proba(self.evaluation[cols])[:,1]
        stored = pd.read_csv(self.reports / "triage_test_predictions.csv")
        np.testing.assert_allclose(stored.loc[stored.comparison.eq("disputed_fields_removed_sensitivity"), "critical_score"], expected, atol=1e-12, rtol=0)

    def test_triage_cli_and_extra_labels_do_not_change_score(self):
        row = self.evaluation.iloc[0].to_dict()
        response = self.triage.predict(row)
        extra = {**row, "severity": "Critical", "incident_type": "unrelated", "is_critical": 1 - row["is_critical"]}
        self.assertEqual(response, self.triage.predict(extra))
        run = cli("score_incident.py", "--model-dir", self.models / "incident_triage", "--incidents", DATA_DIR / "incident_reports.csv", "--incident-id", row["incident_id"])
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr, "")
        self.assertEqual(json.loads(run.stdout), response)
        self.assertFalse(response["score_is_calibrated"])
        self.assertFalse(response["within_fitting_period"])

    def test_incident_input_errors_are_actionable(self):
        run = cli("score_incident.py", "--model-dir", self.models / "incident_triage", "--incidents", DATA_DIR / "incident_reports.csv", "--incident-id", "absent")
        self.assertEqual(run.returncode, 2)
        self.assertIn("exactly one source record", run.stderr)
        row = self.evaluation.iloc[0].to_dict(); row.pop("cpu_utilization")
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            self.triage.predict(row)

    def test_context_excludes_current_value_and_uses_real_history(self):
        row = 9000
        actual = self.features.loc[self.history.timestamp.iloc[row]]
        for col in LOAD_METRICS:
            expected = self.history[col].iloc[row] - self.history[col].iloc[row-60:row].mean()
            self.assertAlmostEqual(actual[col+"__minus_prior_60_mean"], expected, places=9)
        self.assertAlmostEqual(actual["neft_success_rate__non_success_pct"], 100 - self.history.neft_success_rate.iloc[row])
        self.assertEqual(len(self.features), len(self.history)-60)

    def test_anomaly_features_prefix_invariance(self):
        prefix = self.history.iloc[:9001]
        pd.testing.assert_series_equal(anomaly_features(prefix).iloc[-1], self.features.loc[prefix.timestamp.iloc[-1]], atol=1e-9, rtol=0)
        changed = self.history.copy(); changed.loc[9001:, "cpu_utilization"] *= 0.5
        pd.testing.assert_frame_equal(anomaly_features(changed).loc[:prefix.timestamp.iloc[-1]], self.features.loc[:prefix.timestamp.iloc[-1]])

    def test_anomaly_parameters_selected_from_training_stability(self):
        plan = self.selection["anomaly"]
        board = pd.read_csv(self.reports / "anomaly_stability.csv")
        self.assertEqual(plan["candidate_parameters"], list(ANOMALY_CANDIDATES))
        fit = self.features.loc[:pd.Timestamp(plan["initial_fit_through"]) ]
        selection = self.features.loc[plan["selection_start"]:plan["selection_end"]]
        for row in board.to_dict("records"):
            spec = {name:int(row[name]) for name in ("n_estimators", "max_samples")}
            vectors = []
            for offset in plan["seed_offsets"]:
                model = forest(spec, (plan["seed"]+offset) % (2**32))
                with threadpool_limits(limits=1):
                    model.fit(fit); vectors.append(-model.score_samples(selection))
            overlap, correlation = rank_stability(vectors, plan["review_fraction"])
            self.assertAlmostEqual(overlap, row["top_rank_jaccard"], places=10)
            self.assertAlmostEqual(correlation, row["score_spearman"], places=10)
        winner = board.sort_values(["top_rank_jaccard", "score_spearman", "n_estimators", "max_samples"], ascending=[False,False,True,True]).iloc[0]
        self.assertEqual(plan["selected_parameters"], {k:int(winner[k]) for k in ("n_estimators","max_samples")})

    def test_fit_selection_calibration_and_test_are_ordered(self):
        plan = self.selection["anomaly"]
        self.assertLess(pd.Timestamp(plan["initial_fit_through"]), pd.Timestamp(plan["selection_start"]))
        self.assertLess(pd.Timestamp(plan["final_fit_through"]), pd.Timestamp(plan["calibration_start"]))
        self.assertEqual(pd.Timestamp(plan["selection_end"]), pd.Timestamp(plan["final_fit_through"]))
        self.assertEqual(pd.Timestamp(plan["calibrated_through"]), self.cutoff)
        self.assertGreater(self.test.timestamp.iloc[0], self.cutoff)

    def test_calibration_threshold_and_reference_fit_recompute(self):
        model = self.anomaly.model
        plan = self.selection["anomaly"]
        fit = self.features.loc[:pd.Timestamp(plan["final_fit_through"])]
        calibration = self.features.loc[plan["calibration_start"]:plan["calibrated_through"]]
        np.testing.assert_allclose(model.reference_median, fit.median(), atol=1e-10)
        scores = model.raw_scores(calibration)
        self.assertAlmostEqual(model.threshold, np.quantile(scores, .99, method="higher"), places=12)
        self.assertEqual(model.threshold, review_threshold(scores, model.review_fraction))
        self.assertEqual(model.forest.contamination, "auto")
        self.assertEqual(len(fit), plan["final_fit_rows"])

    def test_reported_anomaly_scores_and_counts_recompute(self):
        features = self.features.loc[self.features.index > self.cutoff]
        expected = self.anomaly.model.assess(features)
        saved = pd.read_csv(self.reports / "anomaly_test_scores.csv", parse_dates=["timestamp"])
        np.testing.assert_allclose(expected.anomaly_score, saved.anomaly_score, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(expected.forest_flag, saved.forest_flag)
        np.testing.assert_array_equal(saved.forest_flag, saved.anomaly_score > saved.threshold)
        self.assertEqual(alert_summary(expected), self.summary["anomaly"]["test"])
        self.assertEqual(len(saved), len(self.test))

    def test_constant_feature_guard_uses_observed_uptime_change(self):
        # The real uptime drop lies in calibration, not in the evaluation interval.
        actual = self.features.loc[self.features.service_uptime__loss_pct.gt(0)]
        self.assertGreater(len(actual), 0)
        result = self.anomaly.model.assess(actual)
        self.assertTrue(result.constant_feature_change.all())
        self.assertTrue(result.review_required.all())
        self.assertEqual(self.anomaly.model.constant_values["service_uptime__loss_pct"], 0)

    def test_anomaly_cli_loads_saved_detector(self):
        origin = str(self.test.timestamp.iloc[0])
        expected = self.anomaly.predict(self.history, as_of=origin)
        run = cli("score_anomalies.py", "--model-dir", self.models / "monitoring_anomaly",
                  "--history", DATA_DIR / "infrastructure_metrics_train.csv", "--history", DATA_DIR / "infrastructure_metrics_test.csv", "--as-of", origin)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr, "")
        self.assertEqual(json.loads(run.stdout), expected)
        self.assertEqual(expected["history_rows_used"], 61)
        self.assertFalse(expected["deviations_are_forest_attributions"])

    def test_anomaly_inference_ignores_future_values(self):
        origin = self.test.timestamp.iloc[100]
        expected = self.anomaly.predict(self.history, as_of=str(origin))
        prefix = self.history.loc[self.history.timestamp.le(origin)]
        self.assertEqual(expected, self.anomaly.predict(prefix, as_of=str(origin)))
        changed = self.history.copy(); changed.loc[changed.timestamp.gt(origin), "cpu_utilization"] = np.nan
        self.assertEqual(expected, self.anomaly.predict(changed, as_of=str(origin)))

    def test_retroactive_and_unobserved_origins_rejected(self):
        for origin in (str(self.cutoff), str(self.test.timestamp.iloc[0]+pd.Timedelta(seconds=10))):
            with self.assertRaises(ValueError):
                self.anomaly.predict(self.history, as_of=origin)
        with self.assertRaisesRegex(ValueError, "61 consecutive"):
            self.anomaly.predict(self.test.tail(60))

    def test_invalid_monitoring_values_and_time_grid_rejected(self):
        for col, value in (("cpu_utilization", 101), ("neft_success_rate", -1), ("memory_usage", np.nan),
                           ("network_throughput_mbps", np.inf), ("application_error_rate", 1.5)):
            bad = self.test.tail(61).copy(); bad.iloc[-1,bad.columns.get_loc(col)] = value
            with self.assertRaises(ValueError):
                self.anomaly.predict(bad)
        for bad in (self.test.drop(index=1), self.test.iloc[::-1], pd.concat([self.test, self.test.tail(1)])):
            with self.assertRaises(ValueError):
                anomaly_features(bad)
        aware = self.test.copy(); aware.timestamp = aware.timestamp.dt.tz_localize("UTC")
        with self.assertRaises(ValueError):
            self.anomaly.predict(aware)

    def test_review_budget_validation(self):
        for bad in (0, 0.5, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                review_threshold([0,1,2], bad)
        scores = pd.read_csv(self.reports / "anomaly_calibration.csv").anomaly_score
        self.assertGreaterEqual(review_threshold(scores, .005), review_threshold(scores, .02))

    def test_artifact_checksum_mismatch_refused(self):
        copy = self.directory / "broken"
        shutil.copytree(self.models / "incident_triage", copy)
        with (copy / "model.joblib").open("ab") as handle:
            handle.write(b"damage")
        with self.assertRaisesRegex(ValueError, "checksum"):
            IncidentTriage(copy)

    def test_artifact_runtime_and_metadata_mismatch_refused(self):
        for name in ("runtime", "payload"):
            copy = self.directory / name
            shutil.copytree(self.models / "monitoring_anomaly", copy)
            metadata = json.loads((copy / "metadata.json").read_text())
            if name == "runtime":
                metadata["runtime_versions"]["scikit-learn"] = "0.0.0"
                match = "version mismatch"
            else:
                metadata["plan"]["threshold"] += .1
                match = "metadata disagree"
            write_json(copy / "metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, match):
                MonitoringDetector(copy)

    def test_artifact_kind_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            IncidentTriage(self.models / "monitoring_anomaly")

    def test_existing_outputs_are_not_overwritten(self):
        original = file_sha256(self.models / "incident_triage/model.joblib")
        run = cli("train_detection.py", "--data-dir", DATA_DIR, "--model-dir", self.models, "--report-dir", self.reports)
        self.assertEqual(run.returncode, 2)
        self.assertIn("already exists", run.stderr)
        self.assertEqual(file_sha256(self.models / "incident_triage/model.joblib"), original)

    def test_missing_inputs_and_unsafe_outputs_fail(self):
        run = cli("train_detection.py", "--data-dir", self.directory / "missing", "--model-dir", self.directory / "m", "--report-dir", self.directory / "r")
        self.assertEqual(run.returncode, 2)
        self.assertIn("Missing input", run.stderr)
        run = cli("train_detection.py", "--data-dir", DATA_DIR, "--model-dir", DATA_DIR / "new-models", "--report-dir", self.directory / "r")
        self.assertEqual(run.returncode, 2)
        self.assertIn("separate from", run.stderr)

    def test_full_rerun_reproduces_reports_and_predictions(self):
        model2, report2 = self.directory / "models2", self.directory / "reports2"
        run = cli("train_detection.py", "--data-dir", DATA_DIR, "--model-dir", model2, "--report-dir", report2)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr, "")
        for name in self.reports.iterdir():
            self.assertEqual(name.read_bytes(), (report2/name.name).read_bytes(), name.name)
        self.assertEqual(self.triage.predict(self.evaluation.iloc[0].to_dict()), IncidentTriage(model2/"incident_triage").predict(self.evaluation.iloc[0].to_dict()))
        self.assertEqual(self.anomaly.predict(self.history), MonitoringDetector(model2/"monitoring_anomaly").predict(self.history))

    def test_no_detection_accuracy_claim_for_unlabeled_monitoring(self):
        self.assertFalse({"f1", "precision", "recall", "false_positive_rate", "accuracy"} & set(self.summary["anomaly"]["test"]))
        self.assertIn("no verified event labels", self.train_stdout)

    def test_notebook_code_compiles_and_has_no_committed_outputs(self):
        notebook = json.loads((ROOT / "notebooks/03_incident_detection.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertFalse(cell["outputs"])
                self.assertIsNone(cell["execution_count"])


def main() -> int:
    global DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args(); DATA_DIR = args.data_dir.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DetectionTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
