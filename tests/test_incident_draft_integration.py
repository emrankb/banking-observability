#!/usr/bin/env python3
"""Compare imported draft citations with the saved models and original records."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.api import create_app
from banking_observability.incident_assistant import (
    briefing_from_context, check_generated_reply, prepare_context,
)
from banking_observability.incident_evidence import assemble_investigation

ORIGIN = '2025-10-25T18:01:00'
QUESTION = 'network timeout latency'


class DraftIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [ROOT/'models/knowledge/manifest.json',
                    ROOT/'models/log_clustering/metadata.json',
                    ROOT/'models/detection/monitoring_anomaly/metadata.json']
        missing = [p.parent.name for p in required if not p.is_file()]
        if missing:
            raise RuntimeError('Saved incident-model artifacts are required: ' + ', '.join(missing))
        packet = assemble_investigation(
            ROOT/'data/raw', question=QUESTION, as_of=ORIGIN,
            index_dir=ROOT/'models/knowledge',
            log_model_dir=ROOT/'models/log_clustering',
            anomaly_model_dir=ROOT/'models/detection/monitoring_anomaly',
        )
        cls.context = prepare_context(packet)
        cls.client = TestClient(create_app(ROOT))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def draft(self):
        metrics = [x for x in self.context['source_items'] if x['kind'] == 'monitoring']
        logs = [x for x in self.context['source_items'] if x['kind'] == 'log']
        runbooks = [x for x in self.context['source_items'] if x['kind'] == 'runbook']
        self.assertTrue(metrics and logs and runbooks)
        raw = {
            'summary': 'The recorded minute includes monitoring and log evidence for review.',
            'observations': [{'text': metrics[0]['text'], 'evidence_ids': [metrics[0]['id']]}],
            'hypotheses': [],
            'next_checks': [{'text': 'Review the retrieved runbook passage.',
                             'evidence_ids': [runbooks[0]['id']]}],
            'uncertainty': 'The cause and resolution are not verified by these records.',
        }
        return {**briefing_from_context(self.context), 'status': 'model_draft_with_source_links',
                'model_explanation': deepcopy(check_generated_reply(raw, self.context))}

    def request(self, draft, *, as_of=ORIGIN, question=QUESTION):
        return self.client.post('/api/v1/investigations/check-draft', json={
            'as_of': as_of, 'question': question, 'draft': draft,
        })

    def test_valid_draft_returns_actual_references(self):
        response = self.request(self.draft())
        self.assertEqual(response.status_code, 200, response.text)
        reply = response.json()
        self.assertEqual(reply['status'], 'citation_ids_checked')
        self.assertEqual(reply['model_explanation']['observations'][0]['source_references'][0]['csv_row'], 8283)
        ref = reply['model_explanation']['next_checks'][0]['source_references'][0]
        self.assertEqual(ref['path'], 'knowledge/runbooks.md')
        self.assertTrue(ref['line_end'] >= ref['line_start'])
        self.assertNotIn('root_cause', reply)

    def test_changed_evidence_coordinate_rejected(self):
        draft = self.draft()
        draft['model_explanation']['observations'][0]['source_references'][0]['csv_row'] = 1
        response = self.request(draft)
        self.assertEqual(response.status_code, 422)

    def test_wrong_minute_rejected(self):
        response = self.request(self.draft(), as_of='2025-10-25T18:02:00')
        self.assertEqual(response.status_code, 422)

    def test_unknown_evidence_id_rejected(self):
        draft = self.draft()
        draft['model_explanation']['next_checks'][0]['evidence_ids'] = ['runbook:not-present']
        response = self.request(draft)
        self.assertEqual(response.status_code, 422)

    def test_model_parameters_cannot_be_supplied(self):
        request = {'as_of': ORIGIN, 'question': QUESTION, 'draft': self.draft(),
                   'model': 'some-model', 'allow_external_data': True}
        response = self.client.post('/api/v1/investigations/check-draft', json=request)
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main(verbosity=2)
