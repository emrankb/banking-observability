#!/usr/bin/env python3
"""Check the Docker contract and source-backed deployment smoke assertions."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.api import create_app
from scripts.check_container_ready import REQUIRED_SOURCE_FILES
from scripts.smoke_deployment import (
    ORIGIN, SmokeFailure, local_origin, source_minute, verify_metrics, verify_reference,
)


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose = yaml.safe_load((ROOT / 'compose.yaml').read_text())
        cls.services = cls.compose['services']
        cls.data = ROOT / 'data/raw'

    def test_two_services_use_one_reproducible_image(self):
        self.assertEqual(set(self.services), {'api', 'dashboard'})
        for service in self.services.values():
            self.assertEqual(service['image'], 'banking-observability:local')
            self.assertEqual(service['build'], {'context': '.', 'dockerfile': 'Dockerfile'})

    def test_both_ports_bind_only_to_localhost(self):
        self.assertEqual(self.services['api']['ports'],
                         ['127.0.0.1:${BANKING_API_PORT:-18000}:8000'])
        self.assertEqual(self.services['dashboard']['ports'],
                         ['127.0.0.1:${BANKING_DASHBOARD_PORT:-18501}:8501'])

    def test_data_and_model_mounts_are_read_only_and_not_created(self):
        for name in ('api', 'dashboard'):
            mounts = self.services[name]['volumes']
            self.assertEqual({m['target'] for m in mounts}, {'/app/data/raw', '/app/models'})
            self.assertEqual({m['source'] for m in mounts}, {'./data/raw', './models'})
            self.assertTrue(all(m['type'] == 'bind' and m['read_only'] and
                                m['bind']['create_host_path'] is False for m in mounts))

    def test_containers_use_nonroot_read_only_runtime(self):
        for service in self.services.values():
            self.assertEqual(service['user'], '10001:10001')
            self.assertTrue(service['read_only'])
            self.assertIn('ALL', service['cap_drop'])
            self.assertIn('no-new-privileges:true', service['security_opt'])
            self.assertTrue(any(t.startswith('/tmp:') for t in service['tmpfs']))

    def test_dashboard_waits_for_api_healthy(self):
        depends = self.services['dashboard']['depends_on']
        self.assertEqual(depends['api']['condition'], 'service_healthy')
        for name in ('api', 'dashboard'):
            command = self.services[name]['healthcheck']['test']
            self.assertEqual(command[:3], ['CMD', 'python', 'scripts/check_container_ready.py'])
            self.assertEqual(command[3], name)

    def test_network_binding_in_container_is_separate_from_host(self):
        api = self.services['api']['command']
        dashboard = self.services['dashboard']['command']
        self.assertIn('banking_observability.api:app', api)
        self.assertEqual(api[api.index('--host') + 1], '0.0.0.0')
        self.assertIn('--server.address=0.0.0.0', dashboard)
        self.assertIn('--server.port=8501', dashboard)
        self.assertEqual(self.services['dashboard']['environment']['BANKING_API_URL'], 'http://api:8000')

    def test_external_model_access_not_configured(self):
        for service in self.services.values():
            self.assertNotIn('OPENAI_API_KEY', service['environment'])
            self.assertNotIn('--allow-external-data', ' '.join(service['command']))

    def test_image_matches_artifact_python_patch_and_native_library(self):
        dockerfile = (ROOT / 'Dockerfile').read_text()
        self.assertIn('FROM python:3.13.18-slim-bookworm', dockerfile)
        self.assertIn('libgomp1', dockerfile)
        self.assertIn('python -m pip check', dockerfile)
        self.assertIn('USER banking', dockerfile)
        self.assertNotIn('COPY . .', dockerfile)

    def test_build_context_does_not_include_models_data_or_secrets(self):
        dockerfile = (ROOT / 'Dockerfile').read_text()
        ignored = (ROOT / '.dockerignore').read_text().splitlines()
        for path in ('data/', 'models/', 'reports/', '.venv/', '.git/', '.env', '*.key', '*.joblib'):
            self.assertIn(path, ignored)
        self.assertEqual(ignored[1], '*')
        for source in ('banking_observability/', 'scripts/', 'config/', 'knowledge/', '.streamlit/'):
            self.assertIn(f'COPY {source}', dockerfile)
        self.assertNotIn('COPY data/', dockerfile)
        self.assertNotIn('COPY models/', dockerfile)

    def test_runtime_requirements_include_pinned_model_packages(self):
        reqs = (ROOT / 'requirements-dashboard.txt').read_text()
        self.assertIn('requirements-api.txt', reqs)
        for text, requirement in [('fastapi==0.128.2', 'requirements-api.txt'),
                                  ('faiss-cpu==1.13.2', 'requirements-retrieval.txt'),
                                  ('langgraph==1.2.20', 'requirements-assistant.txt'),
                                  ('numpy==2.3.5', 'requirements-forecast.txt')]:
            self.assertIn(text, (ROOT / requirement).read_text())

    def test_local_service_urls_never_accept_external_addresses(self):
        self.assertEqual(local_origin('http://127.0.0.1:8000/'), 'http://127.0.0.1:8000')
        self.assertEqual(local_origin('http://localhost:8501'), 'http://localhost:8501')
        for url in ('http://api:8000', 'https://localhost:8000', 'http://127.0.0.1',
                    'http://127.0.0.1:8000/metrics', 'http://x@localhost:8000',
                    'http://127.0.0.1:8000?key=1', 'http://0.0.0.0:8000'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                local_origin(url)

    def test_source_file_contract_lists_required_artifacts(self):
        expected = {'infrastructure_metrics.csv', 'infrastructure_metrics_train.csv',
                    'application_logs.csv', 'incident_reports.csv', 'incident_knowledge_base.txt'}
        self.assertEqual(set(REQUIRED_SOURCE_FILES), expected)
        for name in REQUIRED_SOURCE_FILES:
            self.assertTrue((self.data / name).is_file(), name)

    def test_recorded_minute_is_original_csv_row(self):
        source_index, reading = source_minute(self.data)
        self.assertEqual(source_index, 8283)
        self.assertEqual(float(reading['application_error_rate']), 6.0)
        self.assertEqual(float(reading['avg_transaction_processing_time_ms']), 115.21)

    def test_smoke_metric_validation_uses_actual_http_response(self):
        with TestClient(create_app(ROOT)) as client:
            payload = client.get('/api/v1/metrics', params={'start': ORIGIN, 'end': ORIGIN, 'limit': 1})
        self.assertEqual(payload.status_code, 200, payload.text)
        _, source = source_minute(self.data)
        verify_metrics(payload.json(), source)

    def test_smoke_rejects_a_corrupted_real_source_value(self):
        with TestClient(create_app(ROOT)) as client:
            payload = client.get('/api/v1/metrics', params={'start': ORIGIN, 'end': ORIGIN, 'limit': 1}).json()
        _, source = source_minute(self.data)
        changed = dict(source)
        changed['application_error_rate'] = str(float(source['application_error_rate']) + 1)
        with self.assertRaisesRegex(SmokeFailure, 'differs from source'):
            verify_metrics(payload, changed)

    def test_source_citation_must_match_real_file_bytes(self):
        filename = ROOT / 'knowledge/runbooks.md'
        ref = {'path': 'knowledge/runbooks.md',
               'sha256': hashlib.sha256(filename.read_bytes()).hexdigest(),
               'line_start': 7, 'line_end': 13}
        verify_reference(ROOT, ref)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'knowledge').mkdir()
            (folder / 'knowledge/runbooks.md').write_bytes(filename.read_bytes() + b'\n')
            with self.assertRaisesRegex(SmokeFailure, 'checksum'):
                verify_reference(folder, ref)

    def test_source_reference_cannot_traverse_out_of_project(self):
        with self.assertRaisesRegex(SmokeFailure, 'not repository-relative'):
            verify_reference(ROOT, {'path': '../secret.txt', 'sha256': '', 'csv_row': 2})


if __name__ == '__main__':
    unittest.main(verbosity=2)
