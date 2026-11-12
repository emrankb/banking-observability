#!/usr/bin/env python3
"""Check imported-draft validation using the original monitoring CSV."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.incident_assistant import briefing_from_context, check_generated_reply
from banking_observability.incident_drafts import MAX_DRAFT_BYTES, read_draft_json, validate_imported_draft
from banking_observability.incident_evidence import read_evidence

DATA = ROOT / 'data/raw'


class DraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        observed = read_evidence(DATA, as_of='2025-10-25 18:01:00')
        metric = observed['monitoring'][0]
        ref = metric['reference']
        cls.metric = metric
        cls.context = {
            'question': 'network timeout latency', 'as_of': observed['as_of'],
            'review_required': True,
            'source_items': [{
                'id': f"metric:{metric['metric']}", 'kind': 'monitoring', 'as_of': observed['as_of'],
                'text': f"{metric['metric']}={metric['observed']:.4f}; previous_60_minute_median={metric['prior_60_minute_median']:.4f}",
                'source': {'path': ref['path'], 'csv_row': ref['csv_row'], 'sha256': ref['sha256']},
            }],
            'caveats': ['Historical records alone do not identify the cause.'],
        }
        cls.evidence_id = cls.context['source_items'][0]['id']

    def source_backed_draft(self):
        raw = {
            'summary': 'The recorded monitoring minute has available historical measurements.',
            'observations': [{
                'text': f"The recorded {self.metric['metric']} reading is {self.metric['observed']:.2f}.",
                'evidence_ids': [self.evidence_id],
            }],
            'hypotheses': [], 'next_checks': [],
            'uncertainty': 'This reading does not establish the root cause of any log message.',
        }
        return {**briefing_from_context(self.context), 'status': 'model_draft_with_source_links',
                'model_explanation': deepcopy(check_generated_reply(raw, self.context))}

    def test_valid_draft_recovers_original_source_coordinates(self):
        result = validate_imported_draft(self.source_backed_draft(), self.context)
        self.assertEqual(result['status'], 'citation_ids_checked')
        self.assertEqual(result['as_of'], '2025-10-25T18:01:00')
        row = result['model_explanation']['observations'][0]
        self.assertEqual(row['source_references'][0]['csv_row'], 8283)
        self.assertEqual(row['source_references'][0]['path'], 'data/raw/infrastructure_metrics.csv')
        self.assertIn('human review', result['limitation'])
        self.assertNotIn('root_cause', result)

    def test_imported_draft_is_readable_json(self):
        draft = self.source_backed_draft()
        content = json.dumps(draft).encode('utf-8')
        self.assertEqual(read_draft_json(content), draft)

    def test_rejects_an_evidence_only_briefing(self):
        draft = self.source_backed_draft(); draft['status'] = 'evidence_only'
        with self.assertRaisesRegex(ValueError, 'model-generated'):
            validate_imported_draft(draft, self.context)

    def test_rejects_changed_timestamp(self):
        draft = self.source_backed_draft(); draft['as_of'] = '2025-10-25T18:02:00'
        with self.assertRaisesRegex(ValueError, 'as_of'):
            validate_imported_draft(draft, self.context)

    def test_rejects_changed_question(self):
        draft = self.source_backed_draft(); draft['question'] = 'different investigation'
        with self.assertRaisesRegex(ValueError, 'question'):
            validate_imported_draft(draft, self.context)

    def test_rejects_changed_monitoring_value(self):
        draft = self.source_backed_draft()
        draft['monitoring_evidence'][0]['observed_vs_prior_median'] = 'fabricated reading'
        with self.assertRaisesRegex(ValueError, 'monitoring_evidence'):
            validate_imported_draft(draft, self.context)

    def test_rejects_unverified_status_or_extra_fields(self):
        draft = self.source_backed_draft(); draft['confirmed_root_cause'] = 'network'
        with self.assertRaisesRegex(ValueError, 'fields'):
            validate_imported_draft(draft, self.context)

    def test_rejects_inserted_confirmed_cause(self):
        draft = self.source_backed_draft(); draft['root_cause'] = 'network'
        with self.assertRaisesRegex(ValueError, 'root_cause'):
            validate_imported_draft(draft, self.context)

    def test_rejects_unknown_citation_id(self):
        draft = self.source_backed_draft()
        draft['model_explanation']['observations'][0]['evidence_ids'] = ['metric:invented']
        with self.assertRaisesRegex(ValueError, 'unknown evidence'):
            validate_imported_draft(draft, self.context)

    def test_rejects_forged_citation_path(self):
        draft = self.source_backed_draft()
        draft['model_explanation']['observations'][0]['source_references'][0]['path'] = '/tmp/not-the-source.csv'
        with self.assertRaisesRegex(ValueError, 'source reference'):
            validate_imported_draft(draft, self.context)

    def test_rejects_missing_citation_path(self):
        draft = self.source_backed_draft()
        del draft['model_explanation']['observations'][0]['source_references']
        with self.assertRaisesRegex(ValueError, 'original source'):
            validate_imported_draft(draft, self.context)

    def test_rejects_extra_explanation_key(self):
        draft = self.source_backed_draft()
        draft['model_explanation']['verified_diagnosis'] = 'yes'
        with self.assertRaisesRegex(ValueError, 'model explanation'):
            validate_imported_draft(draft, self.context)

    def test_empty_json_rejected(self):
        with self.assertRaises(ValueError): read_draft_json(b'')

    def test_excessive_json_rejected(self):
        with self.assertRaises(ValueError): read_draft_json(b' ' * (MAX_DRAFT_BYTES + 1))

    def test_duplicate_json_fields_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON'):
            read_draft_json(b'{"question":"A","question":"B"}')

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(ValueError): read_draft_json(b'{"value":NaN}')

    def test_invalid_utf8_rejected(self):
        with self.assertRaises(ValueError): read_draft_json(b'\xff\xfe')

    def test_json_scalar_rejected(self):
        with self.assertRaises(ValueError): read_draft_json(b'42')

    def test_draft_is_not_mutated_by_validation(self):
        draft = self.source_backed_draft(); before = deepcopy(draft)
        validate_imported_draft(draft, self.context)
        self.assertEqual(before, draft)


if __name__ == '__main__':
    unittest.main(verbosity=2)
