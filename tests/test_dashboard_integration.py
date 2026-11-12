#!/usr/bin/env python3
"""Connect dashboard requests to the saved detection, log, retrieval, and graph artifacts."""

from pathlib import Path
import sys
import unittest

import httpx
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.api import create_app
from banking_observability.dashboard import DashboardAPI, DashboardError, evidence_links

ORIGIN = "2025-10-25T18:01:00"


class DashboardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import streamlit
        except ImportError as exc:
            raise RuntimeError("Install requirements-dashboard.txt before the dashboard integration tests.") from exc
        models = ROOT / "models"
        needed = [models / "detection/monitoring_anomaly/metadata.json",
                  models / "detection/incident_triage/metadata.json",
                  models / "log_clustering/metadata.json",
                  models / "knowledge/manifest.json"]
        if not all(p.is_file() for p in needed):
            raise RuntimeError("Saved detection, log, and knowledge artifacts must be present in Codespaces.")
        cls.service = TestClient(create_app(ROOT))
        cls.service.__enter__()

        def route(request: httpx.Request) -> httpx.Response:
            response = cls.service.request(request.method,
                str(request.url.path) + ("?" + request.url.query.decode() if request.url.query else ""),
                content=request.content, headers=dict(request.headers))
            return httpx.Response(response.status_code, content=response.content,
                                  headers=dict(response.headers))

        cls.api = DashboardAPI(transport=httpx.MockTransport(route))

    @classmethod
    def tearDownClass(cls):
        cls.api.close()
        cls.service.__exit__(None, None, None)

    def test_dashboard_app_imports_with_streamlit(self):
        import streamlit_app
        self.assertTrue(callable(streamlit_app.main))

    def test_anomaly_replay_uses_saved_detector(self):
        response = self.api.anomaly(as_of=ORIGIN)
        self.assertEqual(response["timestamp"], ORIGIN)
        self.assertTrue(response["review_required"])
        self.assertGreater(response["anomaly_score"], response["threshold"])

    def test_logs_are_classified_and_time_bounded(self):
        response = self.api.logs(as_of=ORIGIN)
        self.assertEqual(response["matching_source_rows"], 14)
        self.assertTrue(all(r["timestamp"] <= ORIGIN for r in response["rows"]))
        self.assertEqual(response["rows"][0]["event_type"], "network_timeout")
        self.assertNotIn("log_message", response["rows"][0])

    def test_actual_source_log_classifies(self):
        row = pd.read_csv(ROOT / "data/raw/application_logs.csv").iloc[1]
        response = self.api.classify(message=row.log_message, level=row.log_level)
        self.assertEqual(response["event_type"], "neft_gateway_timeout")
        self.assertEqual(response["category_source"], "exact_normalized_template_catalog")

    def test_runbooks_retain_verifiable_citations(self):
        result = self.api.knowledge(query="network timeout latency", as_of=ORIGIN)
        self.assertEqual(result["results"][0]["document_id"], "runbook:network_timeout")
        self.assertEqual(result["results"][0]["references"][0]["path"], "knowledge/runbooks.md")
        self.assertGreater(result["results"][0]["references"][0]["line_start"], 0)

    def test_before_index_cutoff_is_rejected(self):
        with self.assertRaises(DashboardError):
            self.api.knowledge(query="network timeout latency", as_of="2025-10-25T14:22:00")

    def test_incident_replay_is_evidence_only(self):
        result = self.api.investigate(question="network timeout latency", as_of=ORIGIN)
        self.assertEqual(result["status"], "evidence_only")
        self.assertTrue(result["review_required"])
        self.assertEqual(result["monitoring_metrics"], 10)
        self.assertEqual(result["recent_classified_logs"], 12)
        self.assertEqual(result["runbook_passages"], 2)
        self.assertIsNone(result["root_cause"])
        self.assertIsNone(result["resolution"])
        self.assertIsNone(result["model_explanation"])

    def test_incident_references_are_original_sources(self):
        reply = self.api.investigate(question="network timeout latency", as_of=ORIGIN)
        links = evidence_links(reply)
        referenced_count = (len(reply["monitoring_evidence"]) + len(reply["recent_log_evidence"])
                            + len(reply["runbook_evidence"]))
        self.assertEqual(len(links), referenced_count)
        self.assertLessEqual(len(reply["recent_log_evidence"]), reply["recent_classified_logs"])
        self.assertIn({"evidence":"runbook:network_timeout", "file":"knowledge/runbooks.md", "row":None, "lines":"137–143"},links)

    def test_severity_mismatch_is_shown_not_discarded(self):
        logs = self.api.logs(as_of=ORIGIN)["rows"]
        self.assertIn("structured_message_severity_mismatch", logs[0]["quality_flags"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
