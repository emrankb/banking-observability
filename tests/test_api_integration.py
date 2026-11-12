#!/usr/bin/env python3
"""Exercise the API with the saved log, detection, knowledge, and graph artifacts."""
from pathlib import Path
import sys
import unittest

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.api import create_app
from banking_observability.incident_assistant import run_assistant
from banking_observability.log_clustering import LogClassifier

ORIGIN = '2025-10-25T18:01:00'


class APIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models = ROOT/'models'
        required = {
            'monitoring detector': models/'detection/monitoring_anomaly/metadata.json',
            'incident triage': models/'detection/incident_triage/metadata.json',
            'log classifier': models/'log_clustering/metadata.json',
            'FAISS knowledge': models/'knowledge/manifest.json',
        }
        missing = [name for name,path in required.items() if not path.is_file()]
        if missing:
            raise RuntimeError('Train the local artifacts before integration testing: ' + ', '.join(missing))
        from banking_observability.knowledge_retrieval import require_faiss
        require_faiss()
        try:
            import langgraph
        except ImportError as exc:
            raise RuntimeError('Install requirements-api.txt before running integration tests.') from exc
        cls.client = TestClient(create_app(ROOT))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_health_reports_full_local_artifacts(self):
        reply = self.client.get('/api/v1/health')
        self.assertEqual(reply.status_code, 200)
        ready = reply.json()['artifacts_present']
        for value in ready.values():
            if isinstance(value, dict):
                self.assertTrue(all(value.values()))
            else:
                self.assertTrue(value)
        self.assertFalse(reply.json()['external_generation'])

    def test_anomaly_at_recorded_minute_matches_saved_detector(self):
        reply = self.client.post('/api/v1/monitoring/anomaly', json={'as_of': ORIGIN})
        self.assertEqual(reply.status_code, 200, reply.text)
        result = reply.json()
        self.assertEqual(result['timestamp'], ORIGIN)
        self.assertTrue(result['review_required'])
        self.assertGreater(result['anomaly_score'], result['threshold'])

    def test_known_neft_log_is_cataloged_not_caused(self):
        logs = pd.read_csv(ROOT/'data/raw/application_logs.csv')
        msg = logs.iloc[1]
        reply = self.client.post('/api/v1/logs/classify',
                                 json={'message':msg.log_message,'log_level':msg.log_level})
        self.assertEqual(reply.status_code, 200, reply.text)
        result = reply.json()
        expected = LogClassifier(ROOT/'models/log_clustering').predict(msg.log_message, msg.log_level)
        self.assertEqual(result, expected)
        self.assertEqual(result['event_type'], 'neft_gateway_timeout')
        self.assertEqual(result['category_source'], 'exact_normalized_template_catalog')
        self.assertNotIn('log_message', result)

    def test_recent_logs_are_masked_and_time_bounded(self):
        reply = self.client.get('/api/v1/logs/recent',
                                params={'as_of': ORIGIN,'lookback_minutes':30})
        self.assertEqual(reply.status_code, 200, reply.text)
        record = reply.json()
        self.assertEqual(record['matching_source_rows'], 14)
        self.assertLessEqual(len(record['rows']), 20)
        self.assertEqual(record['rows'][0]['event_type'], 'network_timeout')
        self.assertIn('structured_message_severity_mismatch', record['rows'][0]['quality_flags'])
        self.assertTrue(all(row['timestamp'] <= ORIGIN for row in record['rows']))
        self.assertTrue(all('log_message' not in row for row in record['rows']))
        self.assertEqual(record['rows'][0]['csv_row'], 4071)

    def test_neft_knowledge_results_retain_source_lines(self):
        reply = self.client.post('/api/v1/knowledge/search',
                          json={'query':'NEFT transaction gateway timeout'})
        self.assertEqual(reply.status_code, 200, reply.text)
        result = reply.json()
        self.assertEqual(result['results'][0]['document_id'],'runbook:neft_gateway_timeout')
        self.assertTrue(result['results'][0]['references'])
        self.assertEqual(result['method'], 'tfidf_faiss_cosine')

    def test_unrelated_knowledge_query_has_no_match(self):
        reply = self.client.post('/api/v1/knowledge/search',
                          json={'query':'telescope orbital galaxy'})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(reply.json()['status'], 'no_vocabulary_overlap')
        self.assertEqual(reply.json()['results'], [])

    def test_history_incident_is_not_a_live_alert(self):
        reply = self.client.get('/api/v1/incidents/INC000136/triage')
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(reply.json()['scope'], 'supplied_incident_snapshot_only')
        self.assertTrue(reply.json()['result'])

    def test_local_assistant_matches_command_logic(self):
        body={'as_of':ORIGIN,'question':'network timeout latency'}
        reply = self.client.post('/api/v1/investigations',json=body)
        self.assertEqual(reply.status_code, 200, reply.text)
        result=reply.json()
        expected=run_assistant(question=body['question'],as_of=ORIGIN,
            data_dir=ROOT/'data/raw',knowledge_dir=ROOT/'models/knowledge',
            log_model_dir=ROOT/'models/log_clustering',
            anomaly_model_dir=ROOT/'models/detection/monitoring_anomaly',
            model=None,allow_external_data=False)
        self.assertEqual(result,expected)
        self.assertEqual(result['status'],'evidence_only')
        self.assertTrue(result['review_required'])
        self.assertEqual(result['monitoring_metrics'],10)
        self.assertEqual(result['recent_classified_logs'],12)
        self.assertEqual(result['runbook_passages'],2)
        self.assertIsNone(result['root_cause'])
        self.assertIsNone(result['resolution'])
        self.assertIsNone(result['model_explanation'])

    def test_external_model_cannot_be_enabled_through_api(self):
        reply = self.client.post('/api/v1/investigations', json={
            'as_of':ORIGIN,'question':'network timeout latency','model':'gpt-6',
            'allow_external_data':True})
        self.assertEqual(reply.status_code, 422)

    def test_retrieval_rejects_before_cutoff_and_allows_cutoff(self):
        earlier = self.client.post('/api/v1/knowledge/search', json={
            'query':'network timeout latency','as_of':'2025-10-25T14:22:00'})
        self.assertEqual(earlier.status_code, 422, earlier.text)
        at_cutoff = self.client.post('/api/v1/knowledge/search', json={
            'query':'network timeout latency','as_of':'2025-10-25T14:23:00'})
        self.assertEqual(at_cutoff.status_code, 200, at_cutoff.text)
        self.assertEqual(at_cutoff.json()['as_of'], '2025-10-25T14:23:00')
        self.assertEqual(at_cutoff.json()['knowledge_cutoff'], '2025-10-25T14:23:00')

    def test_missing_incident_record_is_rejected(self):
        reply = self.client.get('/api/v1/incidents/INC999999/triage')
        self.assertEqual(reply.status_code, 422)


if __name__ == '__main__':
    unittest.main(verbosity=2)
