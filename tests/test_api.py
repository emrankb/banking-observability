#!/usr/bin/env python3
"""Check the local API against the original monitoring CSV and saved forecasts."""
import json
from pathlib import Path
import sys
import unittest

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.api import create_app, read_metric_range, source_minute
from banking_observability.forecasting import LatencyForecaster
from banking_observability.kpi_forecasting import KPIForecaster, TARGETS


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(ROOT))
        cls.monitoring = pd.read_csv(ROOT / 'data/raw/infrastructure_metrics.csv')

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_health_identifies_historical_replay(self):
        response = self.client.get('/api/v1/health')
        self.assertEqual(response.status_code, 200)
        state = response.json()
        self.assertEqual(state['status'], 'ok')
        self.assertEqual(state['mode'], 'historical_replay')
        self.assertFalse(state['external_generation'])
        self.assertTrue(state['artifacts_present']['latency_forecast'])
        self.assertTrue(all(state['artifacts_present']['kpi_forecasts'].values()))
        self.assertNotIn(str(ROOT.resolve()), response.text)

    def test_metric_range_returns_original_source_readings(self):
        result = self.client.get('/api/v1/metrics?start=2025-10-25T18:00:00&end=2025-10-25T18:01:00')
        self.assertEqual(result.status_code, 200)
        rows = result.json()['rows']
        self.assertEqual(len(rows), 2)
        source = self.monitoring.loc[self.monitoring.timestamp.eq('2025-10-25 18:01:00')].iloc[0]
        self.assertEqual(rows[-1]['metrics']['application_error_rate'], float(source['application_error_rate']))
        self.assertEqual(rows[-1]['metrics']['cpu_utilization'], float(source['cpu_utilization']))
        self.assertEqual(rows[-1]['timestamp'], '2025-10-25T18:01:00')

    def test_limit_uses_latest_records_and_counts_omitted(self):
        result = self.client.get('/api/v1/metrics?limit=2')
        self.assertEqual(result.status_code, 200)
        packet = result.json()
        self.assertEqual(packet['count'], 2)
        self.assertEqual(packet['older_matching_rows_omitted'], len(self.monitoring)-2)
        self.assertEqual(packet['rows'][-1]['timestamp'],
                         pd.Timestamp(self.monitoring.timestamp.iloc[-1]).isoformat())

    def test_metric_range_has_no_interpolation_or_new_values(self):
        time = self.monitoring.timestamp.iloc[5000]
        result = self.client.get('/api/v1/metrics', params={'start': time, 'end': time})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['count'], 1)
        self.assertEqual(result.json()['rows'][0]['metrics']['neft_success_rate'],
                         float(self.monitoring.neft_success_rate.iloc[5000]))

    def test_metrics_fail_on_invalid_or_empty_intervals(self):
        for params in ({'start': '2025-10-25T18:02:00', 'end': '2025-10-25T18:01:00'},
                       {'start': '2030-01-01T00:00:00'},
                       {'start': '2025-10-25T18:01:00+00:00'},
                       {'limit': 721}, {'limit': 0}):
            with self.subTest(params=params):
                response = self.client.get('/api/v1/metrics', params=params)
                self.assertEqual(response.status_code, 422)

    def test_metric_summary_uses_earlier_minutes_only(self):
        response = self.client.get('/api/v1/metrics/summary', params={'as_of': '2025-10-25T18:01:00'})
        self.assertEqual(response.status_code, 200)
        packet = response.json()
        values = {x['metric']: x for x in packet['metrics']}
        self.assertEqual(packet['baseline_minutes'], 60)
        self.assertEqual(values['application_error_rate']['observed'], 6.0)
        self.assertEqual(values['application_error_rate']['prior_60_minute_median'], 2.0)
        self.assertEqual(values['cpu_utilization']['reference']['csv_row'], 8283)

    def test_latency_prediction_matches_saved_model(self):
        origin = '2025-10-25T14:24:00'
        response = self.client.post('/api/v1/forecasts/latency', json={'as_of': origin})
        self.assertEqual(response.status_code, 200, response.text)
        packet = response.json()
        expected = LatencyForecaster(ROOT / 'models/latency_5min').predict(self.monitoring, as_of=origin)
        self.assertEqual(packet, expected)
        self.assertEqual(packet['horizon_minutes'], 5)
        self.assertEqual(packet['target_timestamp'], '2025-10-25T14:29:00')
        self.assertNotIn('actual', packet)

    def test_all_six_kpi_models_are_accessible(self):
        origin = '2025-10-25T14:24:00'
        for target in TARGETS:
            with self.subTest(target=target):
                response = self.client.post('/api/v1/forecasts/kpi', json={'as_of': origin, 'target': target})
                self.assertEqual(response.status_code, 200, response.text)
                packet = response.json()
                self.assertEqual(packet,
                    KPIForecaster(ROOT/'models/kpi_5min'/target).predict(self.monitoring, as_of=origin))
                self.assertNotIn('actual', packet)
                self.assertEqual(packet['target_column'], target)

    def test_forecast_rejects_training_interval(self):
        for route, body in [
            ('/api/v1/forecasts/latency', {'as_of': '2025-10-25T14:23:00'}),
            ('/api/v1/forecasts/kpi', {'as_of': '2025-10-25T14:23:00', 'target': 'cpu_utilization'}),
        ]:
            with self.subTest(route=route):
                self.assertEqual(self.client.post(route, json=body).status_code, 422)

    def test_forecast_rejects_non_source_clock(self):
        for value in ['2025-10-25T14:24:10', '2025-10-25T14:24:00Z', 'not-a-date']:
            with self.subTest(value=value):
                self.assertEqual(self.client.post('/api/v1/forecasts/latency',
                                                 json={'as_of': value}).status_code, 422)

    def test_request_cannot_override_model_or_source_paths(self):
        for path, body in [
            ('/api/v1/forecasts/latency', {'as_of': '2025-10-25T14:24:00', 'model_dir': '/tmp/other'}),
            ('/api/v1/forecasts/kpi', {'as_of': '2025-10-25T14:24:00', 'target': 'cpu_utilization', 'history': '/tmp/other'}),
            ('/api/v1/investigations', {'as_of': '2025-10-25T18:01:00', 'question': 'network timeout',
                                        'model': 'remote-model', 'allow_external_data': True}),
        ]:
            with self.subTest(path=path):
                self.assertEqual(self.client.post(path, json=body).status_code, 422)

    def test_kpi_target_is_restricted(self):
        response = self.client.post('/api/v1/forecasts/kpi',
                                    json={'as_of': '2025-10-25T14:24:00', 'target': 'not_a_metric'})
        self.assertEqual(response.status_code, 422)

    def test_knowledge_query_rejects_invalid_threshold(self):
        self.assertEqual(self.client.post('/api/v1/knowledge/search',
                          json={'query': 'timeout', 'min_similarity': 1.2}).status_code, 422)
        self.assertEqual(self.client.post('/api/v1/knowledge/search',
                          json={'query': 'timeout', 'top_k': 100}).status_code, 422)

    def test_log_message_input_rejects_invalid_level_and_blank(self):
        for body in ({'log_level': 'FATAL', 'message': 'ERROR: Something happened'},
                     {'log_level': 'ERROR', 'message': ''}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/api/v1/logs/classify', json=body).status_code, 422)

    def test_invalid_incident_id_rejected_before_model_loading(self):
        self.assertEqual(self.client.get('/api/v1/incidents/../../triage').status_code, 404)
        self.assertEqual(self.client.get('/api/v1/incidents/WRONG/triage').status_code, 422)

    def test_missing_model_is_service_unavailable(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(Path(tmp))) as client:
            response = client.post('/api/v1/forecasts/kpi',
                        json={'as_of':'2025-10-25T14:24:00', 'target':'cpu_utilization'})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(str(Path(tmp).resolve()), response.text)

    def test_openapi_exposes_only_local_evidence_route(self):
        schema = self.client.get('/openapi.json').json()
        self.assertIn('/api/v1/investigations', schema['paths'])
        self.assertNotIn('/api/v1/generate', schema['paths'])
        self.assertIn('/api/v1/metrics', schema['paths'])

    def test_source_minute_rejects_timezone_and_partial_seconds(self):
        for value in ('2025-10-25T18:01:01', '2025-10-25T18:01:00+00:00', 'not-a-date'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                source_minute(value)

    def test_empty_interval_underlying_source_not_modified(self):
        path = ROOT/'data/raw/infrastructure_metrics.csv'
        before = path.stat().st_size
        result = read_metric_range(path, start='2025-10-25T18:01:00', end='2025-10-25T18:01:00', limit=1)
        self.assertEqual(result['count'], 1)
        self.assertEqual(path.stat().st_size, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
