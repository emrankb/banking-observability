#!/usr/bin/env python3
"""Integration checks against fitted models and the built FAISS index."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.incident_evidence import assemble_investigation

DATA = ROOT / "data/raw"
INDEX = ROOT / "models/knowledge"
LOG_MODEL = ROOT / "models/log_clustering"
ANOMALY_MODEL = ROOT / "models/detection/monitoring_anomaly"
ORIGIN = "2025-10-25 18:01:00"
QUESTION = "network timeout latency"


def get_packet(**options):
    return assemble_investigation(DATA, question=QUESTION, as_of=ORIGIN,
                                  index_dir=INDEX, log_model_dir=LOG_MODEL,
                                  anomaly_model_dir=ANOMALY_MODEL, **options)


class InvestigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [INDEX / "manifest.json", LOG_MODEL / "metadata.json", ANOMALY_MODEL / "metadata.json"]
        for name in required:
            if not name.is_file():
                raise FileNotFoundError(f"Missing saved model: {name}. Train its existing checkpoint in the Codespace.")
        cls.packet = get_packet()

    def test_scope_and_no_guessed_root_cause(self):
        answer = self.packet
        self.assertEqual(answer["scope"], "time_bounded_incident_evidence")
        self.assertEqual(answer["interpretation"]["status"], "evidence_only")
        self.assertIsNone(answer["interpretation"]["root_cause"])
        self.assertIsNone(answer["interpretation"]["resolution"])

    def test_existing_detector_scores_the_same_observed_minute(self):
        answer = self.packet
        self.assertEqual(answer["monitoring_detector"]["timestamp"], "2025-10-25T18:01:00")
        self.assertEqual(answer["monitoring_detector"]["history_rows_used"], 61)
        observations = {m["metric"]: m["observed"] for m in answer["observations"]["monitoring"]}
        for metric, value in observations.items():
            self.assertAlmostEqual(value, answer["monitoring_detector"]["observed_metrics"][metric])

    def test_log_labels_are_catalog_labels_not_cluster_labels(self):
        logs = self.packet["observations"]["logs_most_recent_first"]
        self.assertGreater(len(logs), 0)
        for row in logs:
            scored = row["classification"]
            self.assertIn(scored["category_source"], ("exact_normalized_template_catalog", "no_exact_fitted_template"))
            self.assertEqual(scored["normalized_message"], row["normalized_message"])
            self.assertLessEqual(pd.Timestamp(row["timestamp"]), pd.Timestamp(ORIGIN))
        self.assertEqual(logs[0]["classification"]["event_type"], "network_timeout")

    def test_search_provides_source_references(self):
        search = self.packet["runbook_retrieval"]
        self.assertEqual(search["query"], QUESTION)
        self.assertEqual(search["status"], "matches")
        self.assertTrue(search["results"])
        self.assertEqual(search["results"][0]["document_id"], "runbook:network_timeout")
        self.assertTrue(all(row["references"] for row in search["results"]))
        self.assertIsNone(self.packet["historical_incident_retrieval"])

    def test_historical_incident_search_remains_separate(self):
        other = get_packet(include_incident_snapshots=True)
        self.assertEqual(other["historical_incident_retrieval"]["source_kind"], "incident_snapshot")
        self.assertEqual(other["runbook_retrieval"], self.packet["runbook_retrieval"])
        for row in other["historical_incident_retrieval"]["results"]:
            self.assertEqual(row["kind"], "incident_snapshot")
            self.assertLessEqual(pd.Timestamp(row["event_timestamp"]), pd.Timestamp(ORIGIN))

    def test_replay_before_index_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "after the knowledge index"):
            assemble_investigation(DATA, question=QUESTION, as_of="2025-10-25 14:23:00",
                                   index_dir=INDEX, log_model_dir=LOG_MODEL, anomaly_model_dir=ANOMALY_MODEL)

    def test_cli_emits_same_packet(self):
        process = subprocess.run([sys.executable, str(ROOT / "scripts/investigate_incident.py"),
                                  "--as-of", ORIGIN, "--question", QUESTION], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        answer = json.loads(process.stdout)
        self.assertEqual(answer, self.packet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args, rest = parser.parse_known_args()
    DATA = args.data_dir.resolve()
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
