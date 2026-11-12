#!/usr/bin/env python3
"""Check time-bounded evidence against the supplied monitoring and log records."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.incident_evidence import (
    BASELINE_MINUTES, DISPLAY_METRICS, read_evidence, validate_query)
from banking_observability.forecasting import file_sha256

DATA = ROOT / "data/raw"
ORIGIN = "2025-10-25 18:01:00"


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_hashes = {name: file_sha256(DATA / name) for name in
                               ("infrastructure_metrics.csv", "application_logs.csv")}
        cls.packet = read_evidence(DATA, as_of=ORIGIN)
        cls.raw_metrics = pd.read_csv(DATA / "infrastructure_metrics.csv")
        cls.raw_logs = pd.read_csv(DATA / "application_logs.csv")

    @classmethod
    def tearDownClass(cls):
        for name, digest in cls.original_hashes.items():
            if file_sha256(DATA / name) != digest:
                raise AssertionError(f"Original source changed: {name}")

    def test_exact_source_minute_and_prior_history(self):
        p = self.packet
        self.assertEqual(p["as_of"], "2025-10-25T18:01:00")
        self.assertEqual(p["monitoring_rows"], BASELINE_MINUTES + 1)
        self.assertEqual(p["monitoring_window_start"], "2025-10-25T17:01:00")
        self.assertEqual(p["monitoring_reference"]["timestamp"], p["as_of"])

    def test_metric_names_match_existing_contract(self):
        self.assertEqual([m["metric"] for m in self.packet["monitoring"]], list(DISPLAY_METRICS))

    def test_source_monitoring_row_is_exact(self):
        reference = self.packet["monitoring_reference"]
        row = self.raw_metrics.iloc[reference["csv_row"] - 2]
        self.assertEqual(row["timestamp"], ORIGIN)
        self.assertEqual(reference["sha256"], self.original_hashes["infrastructure_metrics.csv"])
        for measure in self.packet["monitoring"]:
            self.assertAlmostEqual(measure["observed"], float(row[measure["metric"]]))
            self.assertEqual(measure["reference"], reference)

    def test_reference_medians_exclude_current_minute(self):
        pos = self.packet["monitoring_reference"]["csv_row"] - 2
        prior = self.raw_metrics.iloc[pos - 60:pos]
        for m in self.packet["monitoring"]:
            expected = prior[m["metric"]].median()
            self.assertAlmostEqual(m["prior_60_minute_median"], float(expected))
            self.assertAlmostEqual(m["difference_from_median"], m["observed"] - expected)

    def test_log_window_is_time_bounded_and_recent_first(self):
        packet = self.packet
        self.assertEqual(packet["returned_log_count"], len(packet["logs_most_recent_first"]))
        self.assertLessEqual(packet["returned_log_count"], 20)
        times = pd.to_datetime([m["timestamp"] for m in packet["logs_most_recent_first"]])
        self.assertTrue(times.is_monotonic_decreasing)
        self.assertTrue((times <= pd.Timestamp(ORIGIN)).all())
        self.assertTrue((times >= pd.Timestamp(ORIGIN) - pd.Timedelta(minutes=30)).all())

    def test_log_rows_and_original_wording_match_source(self):
        for m in self.packet["logs_most_recent_first"]:
            row = self.raw_logs.iloc[m["csv_row"] - 2]
            self.assertEqual(row["log_message"], m["log_message"])
            self.assertEqual(row["log_level"], m["log_level"])
            self.assertEqual(pd.Timestamp(row["timestamp"]).isoformat(), m["timestamp"])
            self.assertEqual(m["reference"]["csv_row"], m["csv_row"])
            self.assertEqual(m["reference"]["sha256"], self.original_hashes["application_logs.csv"])

    def test_exact_known_network_message_is_preserved(self):
        row = next(m for m in self.packet["logs_most_recent_first"] if m["timestamp"] == "2025-10-25T18:01:00")
        self.assertIn("Network timeout", row["log_message"])
        self.assertIn("network timeout", row["normalized_message"])
        self.assertNotIn("srv-05", row["normalized_message"])

    def test_lookback_one_minute_changes_only_log_selection(self):
        narrow = read_evidence(DATA, as_of=ORIGIN, lookback_minutes=1)
        self.assertEqual(narrow["monitoring"], self.packet["monitoring"])
        self.assertLessEqual(narrow["recent_log_count"], self.packet["recent_log_count"])
        self.assertTrue(all(pd.Timestamp(m["timestamp"]) >= pd.Timestamp(ORIGIN) - pd.Timedelta(minutes=1)
                            for m in narrow["logs_most_recent_first"]))

    def test_short_window_returns_only_matching_records(self):
        origin = "2025-10-25 18:00:00"
        packet = read_evidence(DATA, as_of=origin, lookback_minutes=1)
        self.assertLessEqual(packet["recent_log_count"], 2)
        self.assertIsInstance(packet["logs_most_recent_first"], list)

    def test_no_source_changes_when_reading(self):
        for name, digest in self.original_hashes.items():
            self.assertEqual(file_sha256(DATA / name), digest)

    def test_future_metric_changes_do_not_change_evidence_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in self.original_hashes:
                shutil.copy2(DATA / name, directory / name)
            changed = pd.read_csv(directory / "infrastructure_metrics.csv")
            future = changed.index[changed.timestamp.eq("2025-10-25 18:02:00")][0]
            changed.loc[future, "avg_transaction_processing_time_ms"] += 1000
            changed.to_csv(directory / "infrastructure_metrics.csv", index=False)
            after = read_evidence(directory, as_of=ORIGIN)
            for before_metric, later_metric in zip(self.packet["monitoring"], after["monitoring"]):
                for field in ("observed", "prior_60_minute_median", "difference_from_median"):
                    self.assertEqual(before_metric[field], later_metric[field])

    def test_future_log_wording_is_not_normalized_or_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in self.original_hashes:
                shutil.copy2(DATA / name, directory / name)
            changed = pd.read_csv(directory / "application_logs.csv")
            future = changed.index[pd.to_datetime(changed.timestamp).gt(pd.Timestamp(ORIGIN))][0]
            changed.loc[future, "log_message"] = ""  # Future malformed data must not enter an earlier evidence packet.
            changed.to_csv(directory / "application_logs.csv", index=False)
            after = read_evidence(directory, as_of=ORIGIN)
            self.assertEqual([(m["csv_row"], m["log_message"]) for m in after["logs_most_recent_first"]],
                             [(m["csv_row"], m["log_message"]) for m in self.packet["logs_most_recent_first"]])

    def test_missing_monitoring_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one recorded"):
            read_evidence(DATA, as_of="2025-10-28 00:00:00")

    def test_fractional_minute_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "minute boundary"):
            read_evidence(DATA, as_of="2025-10-25 18:01:01")

    def test_early_origin_without_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "60 earlier"):
            read_evidence(DATA, as_of="2025-10-20 00:05:00")

    def test_invalid_lookback_is_rejected(self):
        for value in (-1, 0, 121, True, 0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_evidence(DATA, as_of=ORIGIN, lookback_minutes=value)

    def test_timezone_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone-naive"):
            read_evidence(DATA, as_of="2025-10-25T18:01:00Z")

    def test_missing_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(FileNotFoundError):
            read_evidence(Path(temp), as_of=ORIGIN)

    def test_investigation_questions_require_real_text(self):
        for value in ("", "  ", "x" * 4097):
            with self.subTest(value=value[:10]), self.assertRaises(ValueError):
                validate_query(value)
        self.assertEqual(validate_query("  Check NEFT gateway timeout. "), "Check NEFT gateway timeout.")

    def test_packet_does_not_claim_causes_or_incident_labels(self):
        packet = deepcopy(self.packet)
        self.assertNotIn("root_cause", packet)
        self.assertNotIn("is_critical", packet)
        self.assertNotIn("recommendation", packet)
        self.assertIn("not proof", " ".join(packet["limitations"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args, rest = parser.parse_known_args()
    DATA = args.data_dir.resolve()
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
