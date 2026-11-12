#!/usr/bin/env python3
"""Check cited briefing contracts against the original monitoring CSV."""
import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from banking_observability.incident_evidence import read_evidence
from banking_observability.incident_assistant import (
    response_schema, check_generated_reply, briefing_from_context, run_assistant,
    openai_explanation,
    prepare_context,
)

DATA = ROOT / 'data/raw'
ORIGIN = '2025-10-25 18:01:00'


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_evidence(DATA,as_of=ORIGIN)
        cls.metric = cls.source['monitoring'][0]
        cls.mid = 'metric:' + cls.metric['metric']
        cls.context = {
            'question': 'network timeout latency',
            'as_of': cls.source['as_of'],
            'review_required': False,
            'source_items': [{
                'id': cls.mid, 'kind': 'monitoring', 'as_of': cls.source['as_of'],
                'text': f"{cls.metric['metric']}={cls.metric['observed']}",
                'source': cls.metric['reference'],
            }],
            'caveats': ['Historical monitoring values do not identify a cause.'],
        }

    def response(self):
        return {'summary':'The recorded monitoring minute is available.',
                'observations':[{'text':f"The recorded {self.metric['metric']} value is {self.metric['observed']}.",
                                 'evidence_ids':[self.mid]}],
                'hypotheses':[], 'next_checks':[],
                'uncertainty':'The observed value alone does not identify a cause.'}

    def test_observed_reference_is_real(self):
        self.assertEqual(self.context['source_items'][0]['source']['csv_row'],8283)
        self.assertEqual(self.source['logs_most_recent_first'][0]['csv_row'],4071)

    def test_schema_has_no_confirmed_cause_field(self):
        schema=response_schema()
        self.assertEqual(set(schema['properties']),set(schema['required']))
        self.assertNotIn('root_cause',schema['properties'])
        self.assertFalse(schema['additionalProperties'])


    def test_schema_limits_match_local_validation(self):
        schema = response_schema()
        for field in ("observations", "hypotheses", "next_checks"):
            with self.subTest(field=field):
                values = schema["properties"][field]
                self.assertEqual(values["maxItems"], 5)
                references = values["items"]["properties"]["evidence_ids"]
                self.assertEqual(references["minItems"], 1)
                self.assertEqual(references["maxItems"], 4)

    def test_all_claim_lists_reject_more_than_five_entries(self):
        example = self.response()["observations"][0]
        for field in ("observations", "hypotheses", "next_checks"):
            with self.subTest(field=field):
                response = self.response()
                response[field] = [dict(example) for _ in range(6)]
                with self.assertRaisesRegex(ValueError, "no more than five"):
                    check_generated_reply(response, self.context)

    def test_claim_rejects_more_than_four_evidence_ids(self):
        response = self.response()
        response["observations"][0]["evidence_ids"] = [
            self.mid, "metric:a", "metric:b", "metric:c", "metric:d",
        ]
        with self.assertRaisesRegex(ValueError, "1–4 distinct"):
            check_generated_reply(response, self.context)

    def test_remote_request_uses_bounded_schema_without_network(self):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps(self.response()),
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-placeholder"}):
            with patch("openai.OpenAI", return_value=client):
                explanation = openai_explanation(
                    self.context, model="offline-test-model", send_external=True,
                )
        self.assertEqual(explanation, self.response())
        settings = client.responses.create.call_args.kwargs
        self.assertEqual(settings["text"]["format"]["schema"], response_schema())
        self.assertTrue(settings["text"]["format"]["strict"])
        self.assertFalse(settings["store"])
        self.assertIn("at most five", settings["instructions"])
        self.assertIn("one to four distinct", settings["instructions"])

    def test_real_metric_reference_passes_validation(self):
        result=check_generated_reply(self.response(),self.context)
        self.assertEqual(result['observations'][0]['source_references'][0]['csv_row'],8283)

    def test_unknown_reference_is_rejected(self):
        response=self.response()
        response['observations'][0]['evidence_ids']=['metric:not_a_metric']
        with self.assertRaisesRegex(ValueError,'unknown evidence ID'):
            check_generated_reply(response,self.context)

    def test_duplicate_reference_is_rejected(self):
        response=self.response()
        response['observations'][0]['evidence_ids']=[self.mid,self.mid]
        with self.assertRaises(ValueError):check_generated_reply(response,self.context)

    def test_model_cannot_insert_root_cause_field(self):
        response=self.response();response['root_cause']='not established'
        with self.assertRaises(ValueError):check_generated_reply(response,self.context)

    def test_unreferenced_observation_is_rejected(self):
        response=self.response();response['observations'][0]['evidence_ids']=[]
        with self.assertRaises(ValueError):check_generated_reply(response,self.context)

    def test_long_or_blank_claim_is_rejected(self):
        for value in ('', 'x'*421):
            with self.subTest(value_len=len(value)):
                response=self.response();response['observations'][0]['text']=value
                with self.assertRaises(ValueError):check_generated_reply(response,self.context)

    def test_local_briefing_does_not_invent_resolution(self):
        briefing=briefing_from_context(self.context)
        self.assertEqual(briefing['monitoring_metrics'],1)
        self.assertIsNone(briefing['root_cause'])
        self.assertIsNone(briefing['resolution'])
        self.assertIsNone(briefing['model_explanation'])

    def test_remote_permission_is_required_before_any_index_loading(self):
        kw=dict(question='network timeout latency',as_of=ORIGIN,
                data_dir=DATA,knowledge_dir=ROOT/'models/knowledge',
                log_model_dir=ROOT/'models/log_clustering',
                anomaly_model_dir=ROOT/'models/detection/monitoring_anomaly')
        with self.assertRaisesRegex(ValueError,'allow-external-data'):
            run_assistant(**kw,model='some-remote-model')
        with self.assertRaisesRegex(ValueError,'requires --model'):
            run_assistant(**kw,allow_external_data=True)

    def test_packet_shape_is_checked(self):
        with self.assertRaises(ValueError):prepare_context({'scope':'wrong'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    args, rest=parser.parse_known_args();DATA=args.data_dir.resolve()
    unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
