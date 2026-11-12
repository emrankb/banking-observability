#!/usr/bin/env python3
"""Check the draft import endpoint without requiring model artifacts."""
from pathlib import Path
import sys
import unittest

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.api import create_app


class DraftRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(ROOT))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_schema_exposes_draft_checks_but_no_generation(self):
        paths = self.client.get('/openapi.json').json()['paths']
        self.assertIn('/api/v1/investigations/check-draft', paths)
        self.assertNotIn('/api/v1/investigations/generate', paths)

    def test_unknown_external_model_fields_are_forbidden(self):
        response = self.client.post('/api/v1/investigations/check-draft', json={
            'as_of': '2025-10-25T18:01:00', 'question': 'network timeout latency',
            'draft': {}, 'model': 'gpt-6', 'allow_external_data': True,
        })
        self.assertEqual(response.status_code, 422)

    def test_invalid_minute_is_rejected_before_model_access(self):
        response = self.client.post('/api/v1/investigations/check-draft', json={
            'as_of': '2025-10-25T18:01:30', 'question': 'network timeout latency',
            'draft': {},
        })
        self.assertEqual(response.status_code, 422, response.text)

    def test_oversized_draft_is_rejected_before_model_access(self):
        response = self.client.post('/api/v1/investigations/check-draft', json={
            'as_of': '2025-10-25T18:01:00', 'question': 'network timeout latency',
            'draft': {'large': 'x' * 66000},
        })
        self.assertEqual(response.status_code, 422, response.text)

    def test_other_source_paths_are_forbidden(self):
        response = self.client.post('/api/v1/investigations/check-draft', json={
            'as_of': '2025-10-25T18:01:00', 'question': 'network timeout latency',
            'draft': {}, 'model_dir': '/tmp/alternate',
        })
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main(verbosity=2)
