#!/usr/bin/env python3
"""Verify causal features, temporal evaluation, saved artifacts and both CLIs."""
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
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.forecast_experiment import score, temporal_folds  # noqa: E402
from banking_observability.forecast_features import (  # noqa: E402
    FeatureSpec, REQUIRED_COLUMNS, TARGET, build_features, make_supervised, read_history, validate_history,
)
from banking_observability.forecast_models import candidate_specs  # noqa: E402
from banking_observability.forecasting import LatencyForecaster, file_sha256  # noqa: E402

DATA_DIR: Path
REPORT_DIR: Path
MODEL_DIR: Path


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = FeatureSpec()
        cls.train = read_history(DATA_DIR / "infrastructure_metrics_train.csv")
        cls.test = read_history(DATA_DIR / "infrastructure_metrics_test.csv")
        cls.history = pd.concat([cls.train, cls.test], ignore_index=True)
        cls.train_x, cls.train_y = make_supervised(cls.train, cls.spec)
        cls.features = build_features(cls.history, cls.spec)
        cls.predictions = pd.read_csv(REPORT_DIR / "test_predictions.csv", parse_dates=["forecast_origin", "target_timestamp"])
        cls.cv_predictions = pd.read_csv(REPORT_DIR / "cv_predictions.csv", parse_dates=["forecast_origin", "target_timestamp"])
        cls.cv_metrics = pd.read_csv(REPORT_DIR / "cv_fold_metrics.csv")
        cls.metrics = pd.read_csv(REPORT_DIR / "test_metrics.csv")
        cls.selection = json.loads((REPORT_DIR / "selection.json").read_text())
        cls.summary = json.loads((REPORT_DIR / "summary.json").read_text())
        cls.plan = json.loads((REPORT_DIR / "experiment_plan.json").read_text())
        cls.registry = json.loads((MODEL_DIR / "registry.json").read_text())
        cls.model = LatencyForecaster(MODEL_DIR)
        cls.original_hashes = {p.name: file_sha256(p) for p in DATA_DIR.glob("*.csv")}

    def test_exact_source_target_and_feature_contract(self) -> None:
        self.assertEqual(self.plan["target_column"], TARGET)
        self.assertEqual(self.model.metadata["all_feature_columns"], self.features.columns.tolist())
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(pd.read_csv(DATA_DIR / "infrastructure_metrics.csv", nrows=0).columns))
        self.assertFalse(any("target_ms" in c for c in self.features))

    def test_feature_prefix_invariance_without_future_rows(self) -> None:
        for row in (1440, 5000, 9000):
            prefix = build_features(self.history.iloc[:row + 1], self.spec)
            pd.testing.assert_frame_equal(prefix.tail(1), self.features.iloc[[row]], check_exact=False, atol=1e-10, rtol=0)

    def test_future_mutations_cannot_change_features(self) -> None:
        row = 9000
        changed = self.history.copy()
        changed.loc[row + 1:, list(REQUIRED_COLUMNS[1:])] *= 0.5
        other = build_features(changed, self.spec)
        pd.testing.assert_frame_equal(other.iloc[[row]], self.features.iloc[[row]], check_exact=False, atol=1e-10, rtol=0)

    def test_actual_lags_and_inclusive_trailing_windows(self) -> None:
        row = 7000
        for lag in self.spec.latency_lags:
            self.assertEqual(self.features.iloc[row][f"latency__lag_{lag}"], self.history.iloc[row - lag][TARGET])
        for window in self.spec.rolling_windows:
            raw = self.history[TARGET].iloc[row - window + 1:row + 1]
            self.assertAlmostEqual(self.features.iloc[row][f"latency__mean_{window}"], raw.mean(), places=10)
            self.assertAlmostEqual(self.features.iloc[row][f"latency__std_{window}"], raw.std(ddof=0), places=8)

    def test_daily_baseline_is_aligned_to_target_not_origin(self) -> None:
        row = 7000
        expected = self.history.iloc[row + 5 - 1440][TARGET]
        self.assertEqual(self.features.iloc[row]["latency__previous_day_at_target"], expected)

    def test_supervised_target_is_exactly_five_minutes_ahead(self) -> None:
        indexed = self.train.set_index("timestamp")[TARGET]
        expected = indexed.reindex(self.train_y.index + pd.Timedelta(minutes=5)).to_numpy()
        np.testing.assert_array_equal(self.train_y.to_numpy(), expected)
        self.assertEqual(len(self.train_y), len(self.train) - 1440 - 5)

    def test_known_future_calendar_needs_no_future_metrics(self) -> None:
        row = int(self.history.index[self.history.timestamp.eq("2025-10-24 23:57:00")][0])
        self.assertEqual(self.features.iloc[row]["clock__minute_of_day"], 2)
        self.assertEqual(self.features.iloc[row]["clock__is_weekend"], 1)

    def test_every_temporal_fold_purges_overlapping_labels(self) -> None:
        folds = temporal_folds(self.train_x, horizon=5)
        bounds = pd.read_csv(REPORT_DIR / "folds.csv")
        self.assertEqual(len(folds), 3)
        for number, (train, valid) in enumerate(folds):
            self.assertEqual(len(valid), 1440)
            self.assertEqual(valid[0] - train[-1] - 1, 5)
            self.assertLess(self.train_x.index[train[-1]] + pd.Timedelta(minutes=5), self.train_x.index[valid[0]])
            self.assertEqual(pd.Timestamp(bounds.iloc[number].train_target_end), self.train_x.index[train[-1]] + pd.Timedelta(minutes=5))

    def test_validation_actuals_are_from_training_file_only(self) -> None:
        raw = self.train.set_index("timestamp")[TARGET]
        expected = raw.reindex(self.cv_predictions.target_timestamp).to_numpy()
        self.assertTrue(np.isfinite(expected).all())
        np.testing.assert_array_equal(expected, self.cv_predictions.actual_ms)

    def test_all_validation_metrics_recompute_from_predictions(self) -> None:
        for (name, fold), rows in self.cv_predictions.groupby(["candidate", "fold"]):
            expected = score(rows.actual_ms, rows.prediction_ms)
            observed = self.cv_metrics.loc[self.cv_metrics.candidate.eq(name) & self.cv_metrics.fold.eq(fold)].iloc[0]
            for key, value in expected.items():
                self.assertAlmostEqual(float(observed[key]), value, places=9)

    def test_winner_is_chosen_by_validation_not_test_rank(self) -> None:
        board = pd.read_csv(REPORT_DIR / "cv_leaderboard.csv").sort_values(["mean_mae_ms", "mean_rmse_ms", "candidate"])
        winner = self.selection["selected_candidate"]["name"]
        self.assertEqual(winner, board.iloc[0].candidate)
        self.assertEqual(self.model.model.specification["name"], winner)
        self.assertEqual(self.metrics.selected.sum(), 1)
        self.assertEqual(self.metrics.loc[self.metrics.selected, "candidate"].iloc[0], winner)

    def test_candidate_definitions_match_saved_plan(self) -> None:
        self.assertEqual(self.plan["candidates"], candidate_specs())
        self.assertEqual(len(self.cv_metrics), len(candidate_specs()) * 3)
        expected_seed = int(file_sha256(DATA_DIR / "infrastructure_metrics_train.csv")[:8], 16)
        self.assertEqual(self.plan["seed"], expected_seed)

    def test_all_test_metrics_recompute_from_raw_targets_and_predictions(self) -> None:
        actual = self.history.set_index("timestamp")[TARGET].reindex(self.predictions.target_timestamp).to_numpy()
        np.testing.assert_array_equal(actual, self.predictions.actual_ms)
        for row in self.metrics.to_dict("records"):
            recomputed = score(actual, self.predictions[row["candidate"]])
            for key, value in recomputed.items():
                self.assertAlmostEqual(float(row[key]), value, places=9)

    def test_test_replay_alignment_and_coverage(self) -> None:
        self.assertEqual(len(self.predictions), len(self.test) - 5)
        self.assertEqual(self.predictions.forecast_origin.iloc[0], self.test.timestamp.iloc[0])
        self.assertTrue((self.predictions.target_timestamp - self.predictions.forecast_origin).eq(pd.Timedelta(minutes=5)).all())
        self.assertEqual(self.predictions.target_timestamp.iloc[-1], self.test.timestamp.iloc[-1])
        self.assertLess(pd.Timestamp(self.model.metadata["trained_target_through"]), self.predictions.forecast_origin.iloc[0])

    def test_every_saved_artifact_reproduces_its_test_predictions(self) -> None:
        x = self.features.loc[pd.DatetimeIndex(self.predictions.forecast_origin)]
        with threadpool_limits(limits=1):
            for entry in self.registry:
                loaded = LatencyForecaster(MODEL_DIR / entry["path"])
                np.testing.assert_allclose(loaded.model.predict(x), self.predictions[entry["name"]], rtol=0, atol=1e-10)

    def test_inference_cli_for_every_saved_artifact(self) -> None:
        origin = self.test.timestamp.iloc[0]
        for entry in self.registry:
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/predict_forecast.py"),
                "--model-dir", str(MODEL_DIR / entry["path"]),
                "--history", str(DATA_DIR / "infrastructure_metrics.csv"), "--as-of", str(origin),
            ], capture_output=True, text=True, timeout=30, check=True)
            self.assertEqual(result.stderr, "")
            response = json.loads(result.stdout)
            self.assertAlmostEqual(response["prediction_ms"], self.predictions[entry["name"]].iloc[0], places=8)
            self.assertEqual(response["horizon_minutes"], 5)

    def test_cli_can_forecast_beyond_available_observations_without_inventing_actual(self) -> None:
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/predict_forecast.py"), "--model-dir", str(MODEL_DIR),
            "--history", str(DATA_DIR / "infrastructure_metrics_test.csv"),
        ], capture_output=True, text=True, timeout=30, check=True)
        response = json.loads(result.stdout)
        self.assertEqual(pd.Timestamp(response["target_timestamp"]), self.test.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
        self.assertNotIn("actual_ms", response)
        self.assertTrue(np.isfinite(response["prediction_ms"]))

    def test_inference_as_of_is_unchanged_by_future_rows(self) -> None:
        origin = self.test.timestamp.iloc[100]
        prefix = self.history.loc[self.history.timestamp.le(origin)]
        with threadpool_limits(limits=1):
            a = self.model.predict(prefix, as_of=origin)
            b = self.model.predict(self.history, as_of=origin)
        self.assertEqual(a, b)

    def test_missing_columns_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_history(self.train.drop(columns="cpu_utilization"))

    def test_missing_nonfinite_and_nonnumeric_metrics_rejected(self) -> None:
        for value in (np.nan, np.inf, -np.inf, "not-a-number"):
            frame = self.train.iloc[:20].copy()
            frame[TARGET] = frame[TARGET].astype(object)
            frame.loc[5, TARGET] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_history(frame)

    def test_ambiguous_timestamp_histories_rejected(self) -> None:
        duplicate = self.train.iloc[:20].copy()
        duplicate.loc[5, "timestamp"] = duplicate.loc[4, "timestamp"]
        second = self.train.iloc[:20].copy()
        second.loc[5, "timestamp"] += pd.Timedelta(seconds=1)
        aware = self.train.iloc[:20].copy()
        aware["timestamp"] = aware.timestamp.dt.tz_localize("UTC")
        for frame in (duplicate, second, aware, self.train.iloc[:20].iloc[::-1], self.train.iloc[:20].drop(index=5)):
            with self.subTest(first=str(frame.timestamp.iloc[0])), self.assertRaises(ValueError):
                validate_history(frame)

    def test_negative_and_out_of_scope_resource_values_rejected(self) -> None:
        for column, value in ((TARGET, -1), ("cpu_utilization", 101)):
            frame = self.train.iloc[:20].copy()
            frame.loc[5, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_history(frame)

    def test_insufficient_shared_history_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Need at least"):
            self.model.predict(self.test.iloc[:20])

    def test_backcasting_before_training_cutoff_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "training targets"):
            self.model.predict(self.history, as_of=self.train.timestamp.iloc[-1])

    def test_unobserved_origin_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.model.predict(self.history, as_of=self.test.timestamp.iloc[10] + pd.Timedelta(seconds=30))

    def test_corrupt_artifact_is_refused_before_unpickling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)
            shutil.copy(MODEL_DIR / "metadata.json", copy / "metadata.json")
            (copy / "model.joblib").write_bytes((MODEL_DIR / "model.joblib").read_bytes() + b"corruption-test")
            with self.assertRaisesRegex(ValueError, "checksum"):
                LatencyForecaster(copy)

    def test_environment_mismatch_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory)
            shutil.copy(MODEL_DIR / "model.joblib", copy / "model.joblib")
            metadata = dict(self.model.metadata)
            metadata["runtime_versions"] = dict(metadata["runtime_versions"], **{"scikit-learn": "deliberate-test-mismatch"})
            (copy / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                LatencyForecaster(copy)

    def test_tree_models_disable_random_internal_early_stopping(self) -> None:
        for entry in self.registry:
            loaded = LatencyForecaster(MODEL_DIR / entry["path"])
            if loaded.model.specification["kind"] == "hgb":
                self.assertFalse(loaded.model.estimator.early_stopping)
                self.assertFalse(loaded.model.estimator.do_early_stopping_)

    def test_ridge_scaler_was_fit_on_training_features_only(self) -> None:
        for entry in self.registry:
            loaded = LatencyForecaster(MODEL_DIR / entry["path"])
            if loaded.model.specification["kind"] == "ridge":
                scaler = loaded.model.estimator.named_steps["standardscaler"]
                expected = self.train_x[loaded.model.columns].mean().to_numpy()
                np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-12, atol=1e-10)

    def test_training_cli_reproduces_selection_metrics_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forecast-reproduction-") as temporary:
            destination = Path(temporary)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/train_forecast.py"), "--data-dir", str(DATA_DIR),
                "--output-dir", str(destination / "reports"), "--models-dir", str(destination / "models"),
            ], capture_output=True, text=True, timeout=180, check=True)
            self.assertEqual(result.stderr, "")
            for name in ("cv_leaderboard.csv", "test_metrics.csv", "test_predictions.csv"):
                pd.testing.assert_frame_equal(pd.read_csv(REPORT_DIR / name), pd.read_csv(destination / "reports" / name), check_exact=False, rtol=0, atol=1e-10)
            rerun = json.loads((destination / "reports/selection.json").read_text())
            self.assertEqual(rerun, self.selection)

    def test_source_bytes_match_original_and_recorded_fingerprints(self) -> None:
        for name, digest in self.original_hashes.items():
            self.assertEqual(file_sha256(DATA_DIR / name), digest)
        for name, digest in self.summary["source_sha256"].items():
            self.assertEqual(file_sha256(DATA_DIR / name), digest)

    def test_notebook_source_compiles_without_checked_in_outputs(self) -> None:
        notebook = json.loads((ROOT / "notebooks/02_latency_forecasting.ipynb").read_text())
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreater(len(code_cells), 0)
        for cell in code_cells:
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            ast.parse(source)
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell.get("outputs", []), [])

    def test_report_cli_recomputes_metrics(self) -> None:
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/write_forecast_report.py"),
            "--data-dir", str(DATA_DIR), "--report-dir", str(REPORT_DIR),
            "--model-dir", str(MODEL_DIR),
        ], capture_output=True, text=True, timeout=30, check=True)
        self.assertEqual(result.stderr, "")
        report = (REPORT_DIR / "forecast_review.md").read_text()
        chosen = self.metrics.loc[self.metrics.selected].iloc[0]
        self.assertIn(self.selection["selected_candidate"]["name"], report)
        self.assertIn(f"**{chosen.mae_ms:.4f} ms**", report)
        self.assertIn(f"**{chosen.rmse_ms:.4f} ms**", report)
        self.assertIn(self.model.metadata["candidate"]["name"], (MODEL_DIR / "MODEL_CARD.md").read_text())

    def test_report_rejects_changed_source_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forecast-source-check-") as temporary:
            data = Path(temporary)
            for name in ("infrastructure_metrics_train.csv", "infrastructure_metrics_test.csv"):
                shutil.copy2(DATA_DIR / name, data / name)
            with (data / "infrastructure_metrics_train.csv").open("ab") as changed:
                changed.write(b"\n")
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/write_forecast_report.py"),
                "--data-dir", str(data), "--report-dir", str(REPORT_DIR),
                "--model-dir", str(MODEL_DIR),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Source fingerprint mismatch", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_report_rejects_stale_evaluation_scores(self) -> None:
        with tempfile.TemporaryDirectory(prefix="forecast-report-check-") as temporary:
            reports = Path(temporary) / "reports"
            shutil.copytree(REPORT_DIR, reports)
            metrics = pd.read_csv(reports / "test_metrics.csv")
            metrics.loc[0, "mae_ms"] += 1
            metrics.to_csv(reports / "test_metrics.csv", index=False)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/write_forecast_report.py"),
                "--data-dir", str(DATA_DIR), "--report-dir", str(reports),
                "--model-dir", str(MODEL_DIR),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("metrics do not match predictions", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_training_cli_refuses_implicit_overwrite(self) -> None:
        before = {path.name: file_sha256(path) for path in MODEL_DIR.glob("*") if path.is_file()}
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/train_forecast.py"), "--data-dir", str(DATA_DIR),
            "--output-dir", str(REPORT_DIR), "--models-dir", str(MODEL_DIR),
        ], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("checkpoint already exists", result.stderr)
        for name, digest in before.items():
            self.assertEqual(file_sha256(MODEL_DIR / name), digest)

    def test_bad_feature_spec_rejected(self) -> None:
        for args in ({"horizon_minutes": 0}, {"horizon_minutes": 1440}, {"latency_lags": (-1,)}, {"rolling_windows": (0,)}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                FeatureSpec(**args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/forecast")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/latency_5min")
    args = parser.parse_args()
    DATA_DIR, REPORT_DIR, MODEL_DIR = args.data_dir.resolve(), args.report_dir.resolve(), args.model_dir.resolve()
    required = [
        *(DATA_DIR / name for name in ("infrastructure_metrics.csv", "infrastructure_metrics_train.csv", "infrastructure_metrics_test.csv")),
        *(REPORT_DIR / name for name in ("test_predictions.csv", "cv_predictions.csv", "cv_fold_metrics.csv", "test_metrics.csv", "cv_leaderboard.csv", "selection.json", "summary.json", "experiment_plan.json", "folds.csv")),
        MODEL_DIR / "registry.json", MODEL_DIR / "metadata.json", MODEL_DIR / "model.joblib",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.exit(2, "Missing forecast inputs or generated artifacts:\n" + "\n".join(missing)
                    + "\nRun scripts/train_forecast.py with these data, report, and model directories first.\n")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ForecastTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    status = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "successful": result.wasSuccessful()}
    (REPORT_DIR / "test_summary.json").write_text(json.dumps(status, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
