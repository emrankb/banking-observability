#!/usr/bin/env python3
"""Exercise the dashboard adapter against real recorded data and the local API."""
from datetime import date, time

from pathlib import Path
import sys
import unittest

import httpx
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.api import create_app
from banking_observability.dashboard import (
    DashboardAPI, DashboardError, FORECAST_TARGETS, KPI_GROUPS,
    evidence_links, forecast_fields, monitoring_frame, recorded_minute,
    recorded_minutes, source_date_time, summary_by_metric, validate_api_url,
)

ORIGIN = "2025-10-25T18:01:00"
FIRST_TEST = "2025-10-25T14:24:00"


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = TestClient(create_app(ROOT))
        cls.service.__enter__()

        # Route the dashboard's HTTP requests through the actual FastAPI application.
        # Both sides read the original CSVs and saved artifacts, not test fixtures.
        def route(request: httpx.Request) -> httpx.Response:
            response = cls.service.request(
                request.method, str(request.url.path) + (
                    "?" + request.url.query.decode() if request.url.query else ""),
                content=request.content, headers=dict(request.headers),
            )
            return httpx.Response(response.status_code, content=response.content,
                                  headers=dict(response.headers))

        cls.api = DashboardAPI(transport=httpx.MockTransport(route))

    @classmethod
    def tearDownClass(cls):
        cls.api.close()
        cls.service.__exit__(None, None, None)

    def test_source_time_is_naive_whole_minute(self):
        self.assertEqual(recorded_minute(ORIGIN), ORIGIN)
        self.assertEqual(source_date_time(date(2025, 10, 25), time(18, 1)), ORIGIN)

    def test_timezone_partial_and_invalid_dates_rejected(self):
        for value in ("2025-10-25T18:01:11", "2025-10-25T18:01:00Z",
                      "2025-10-25T18:01:00+00:00", "2025-02-30T18:01:00", "2025-10-25"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                recorded_minute(value)

    def test_url_is_operator_configured_plain_http_origin(self):
        self.assertEqual(validate_api_url("http://127.0.0.1:8000/"), "http://127.0.0.1:8000")
        for url in ("https://127.0.0.1:8000", "http://user:secret@127.0.0.1:8000",
                    "http://127.0.0.1:8000/v1", "http://127.0.0.1:8000?token=a"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_api_url(url)

    def test_health_is_historical_and_no_remote_generation(self):
        reply = self.api.health()
        self.assertEqual(reply["status"], "ok")
        self.assertEqual(reply["mode"], "historical_replay")
        self.assertIs(reply["external_generation"], False)

    def test_metric_contract_names_match_source(self):
        all_metrics = [m for cols in KPI_GROUPS.values() for m in cols]
        self.assertEqual(len(all_metrics), len(set(all_metrics)))
        columns = set(pd.read_csv(ROOT / "data/raw/infrastructure_metrics.csv", nrows=0).columns)
        self.assertTrue(set(all_metrics).issubset(columns))

    def test_monitoring_window_end_is_not_exceeded(self):
        raw = self.api.metrics(end=ORIGIN, limit=240)
        self.assertEqual(raw["count"], 240)
        rows = recorded_minutes(raw)
        self.assertEqual(len(rows), 240)
        self.assertEqual(rows[-1], ORIGIN)
        self.assertTrue(all(t <= ORIGIN for t in rows))

    def test_monitoring_frame_matches_unmodified_original(self):
        raw = self.api.metrics(end=ORIGIN, limit=10)
        frame = monitoring_frame(raw)
        source = pd.read_csv(ROOT / "data/raw/infrastructure_metrics.csv")
        src = source.loc[source.timestamp.eq("2025-10-25 18:01:00")].iloc[0]
        self.assertAlmostEqual(frame.iloc[-1]["application_error_rate"], float(src.application_error_rate))
        self.assertAlmostEqual(frame.iloc[-1]["cpu_utilization"], float(src.cpu_utilization))
        self.assertEqual(frame.index[-1].isoformat(), ORIGIN)

    def test_monitoring_frame_rejects_reordered_rows(self):
        source = self.api.metrics(end=ORIGIN, limit=4)
        source["rows"] = source["rows"][::-1]
        with self.assertRaises(ValueError):
            monitoring_frame(source)

    def test_monitoring_frame_rejects_duplicates(self):
        source = self.api.metrics(end=ORIGIN, limit=4)
        source["rows"][1]["timestamp"] = source["rows"][0]["timestamp"]
        with self.assertRaises(ValueError):
            monitoring_frame(source)

    def test_summary_is_recorded_with_causal_reference(self):
        response = self.api.summary(as_of=ORIGIN)
        rows = summary_by_metric(response)
        self.assertAlmostEqual(rows["application_error_rate"]["observed"], 6.0)
        self.assertAlmostEqual(rows["application_error_rate"]["prior_60_minute_median"], 2.0)
        self.assertEqual(rows["application_error_rate"]["reference"]["csv_row"], 8283)

    def test_forecast_model_has_no_future_actual(self):
        result = self.api.forecast(as_of=FIRST_TEST, target="cpu_utilization")
        view = forecast_fields(result)
        self.assertEqual(view["target_time"], "2025-10-25T14:29:00")
        self.assertEqual(view["model"], "trailing_15")
        self.assertAlmostEqual(view["value"], 50.29, places=2)
        self.assertNotIn("actual", result)

    def test_latency_model_uses_milliseconds(self):
        result = self.api.forecast(as_of=FIRST_TEST, target="avg_transaction_processing_time_ms")
        view = forecast_fields(result)
        self.assertEqual(view["unit"], "ms")
        self.assertEqual(view["target_time"], "2025-10-25T14:29:00")
        self.assertGreater(view["value"], 0)

    def test_all_seven_forecast_targets_are_exposed(self):
        self.assertEqual(len(FORECAST_TARGETS), 7)
        self.assertIn("rtgs_success_rate", FORECAST_TARGETS.values())

    def test_unknown_forecast_target_rejected(self):
        with self.assertRaises(ValueError):
            self.api.forecast(as_of=FIRST_TEST, target="future_alert_probability")

    def test_training_period_forecast_is_rejected_by_api(self):
        with self.assertRaises(DashboardError):
            self.api.forecast(as_of="2025-10-25T14:23:00", target="cpu_utilization")

    def test_source_interval_outside_data_is_rejected(self):
        with self.assertRaises(DashboardError):
            self.api.metrics(end="2024-01-01T00:00:00", limit=10)

    def test_invalid_log_level_rejected(self):
        with self.assertRaises(ValueError):
            self.api.classify(message="some recorded log", level="UNKNOWN")

    def test_invalid_history_window_rejected(self):
        with self.assertRaises(ValueError):
            self.api.metrics(end=ORIGIN, limit=721)

    def test_duplicate_metric_summary_rejected(self):
        response = self.api.summary(as_of=ORIGIN)
        response["metrics"].append(response["metrics"][0])
        with self.assertRaises(ValueError):
            summary_by_metric(response)

    def test_log_source_rows_are_not_exposed_by_classification(self):
        # Classify needs an observed level and message, not model labels or a filesystem path.
        with self.assertRaises(ValueError):
            self.api.classify(message=" ", level="ERROR")

    def test_bad_query_rejected_before_api_call(self):
        with self.assertRaises(ValueError):
            self.api.knowledge(query="", as_of=ORIGIN)
        with self.assertRaises(ValueError):
            self.api.investigate(question="", as_of=ORIGIN)

    def test_evidence_references_have_source_coordinates(self):
        from banking_observability.incident_evidence import read_evidence
        recorded = read_evidence(ROOT / "data/raw", as_of=ORIGIN, lookback_minutes=30)
        metric = recorded["monitoring"][0]
        first_log = recorded["logs_most_recent_first"][0]
        packet = {
            "monitoring_evidence": [{"id": f"metric:{metric['metric']}",
                                     "source": metric["reference"]}],
            "recent_log_evidence": [{"id": f"log:{first_log['csv_row']}",
                                     "source": first_log["reference"]}],
            "runbook_evidence": [],
        }
        links = evidence_links(packet)
        self.assertEqual(links[0]["row"], 8283)
        self.assertEqual(links[0]["file"], "data/raw/infrastructure_metrics.csv")
        self.assertEqual(links[1]["row"], 4071)
        self.assertEqual(links[1]["file"], "data/raw/application_logs.csv")



if __name__ == "__main__":
    unittest.main(verbosity=2)
