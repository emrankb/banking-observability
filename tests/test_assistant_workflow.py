#!/usr/bin/env python3
"""Exercise the LangGraph route with the saved Codespaces models."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from banking_observability.incident_assistant import (
    create_workflow,prepare_context,briefing_from_context,run_assistant,check_generated_reply
)
from banking_observability.incident_evidence import assemble_investigation

DATA=ROOT/'data/raw'
ORIGIN='2025-10-25 18:01:00'
QUESTION='network timeout latency'


class AssistantWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kw={
            'data_dir': DATA,'knowledge_dir':ROOT/'models/knowledge',
            'log_model_dir':ROOT/'models/log_clustering',
            'anomaly_model_dir':ROOT/'models/detection/monitoring_anomaly',
        }
        cls.packet=assemble_investigation(DATA,question=QUESTION,as_of=ORIGIN,
             index_dir=cls.kw['knowledge_dir'],log_model_dir=cls.kw['log_model_dir'],
             anomaly_model_dir=cls.kw['anomaly_model_dir'])
        cls.context=prepare_context(cls.packet)

    def test_packet_is_same_recorded_minute(self):
        self.assertEqual(self.context['as_of'],'2025-10-25T18:01:00')
        self.assertTrue(self.context['review_required'])

    def test_source_catalog_keeps_original_refs(self):
        sources={row['id']:row for row in self.context['source_items']}
        self.assertEqual(sources['metric:application_error_rate']['source']['csv_row'],8283)
        self.assertEqual(sources['log:4071']['source']['csv_row'],4071)
        self.assertIn('runbook:network_timeout',sources)
        self.assertEqual(sources['log:4071']['kind'],'log')

    def test_log_context_drops_variable_identifiers(self):
        rows=[r for r in self.context['source_items'] if r['kind']=='log']
        self.assertTrue(rows)
        self.assertIn('<host_id>',rows[0]['text'])
        self.assertNotIn('srv-05',rows[0]['text'])
        self.assertIn('structured_message_severity_mismatch',rows[0]['text'])

    def test_runbook_citations_retain_lines(self):
        note=[r for r in self.context['source_items'] if r['id']=='runbook:network_timeout'][0]
        self.assertEqual(note['source']['path'],'knowledge/runbooks.md')
        self.assertEqual(note['source']['line_start'],137)

    def test_graph_runs_evidence_only_without_external_model(self):
        output=run_assistant(question=QUESTION,as_of=ORIGIN,**self.kw)
        self.assertEqual(output['status'],'evidence_only')
        self.assertTrue(output['review_required'])
        self.assertEqual(output['monitoring_metrics'],10)
        self.assertIsNone(output['model_explanation'])
        self.assertIsNone(output['root_cause'])

    def test_cli_matches_graph(self):
        from subprocess import run
        import json
        cmd=[sys.executable,str(ROOT/'scripts/ask_incident.py'),'--as-of',ORIGIN,'--question',QUESTION]
        proc=run(cmd,text=True,capture_output=True,cwd=ROOT)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['status'],'evidence_only')

    def test_incorrect_generated_citation_is_rejected(self):
        reply={'summary':'Review requested for the observed minute.',
               'observations':[{'text':'The recorded error rate was 6.0.',
                                'evidence_ids':['metric:application_error_rate']}],
               'hypotheses':[], 'next_checks':[],
               'uncertainty':'The cause is unknown.'}
        result=check_generated_reply(reply,self.context)
        self.assertEqual(result['observations'][0]['source_references'][0]['csv_row'],8283)
        reply['observations'][0]['evidence_ids']=['metric:nonexistent']
        with self.assertRaises(ValueError):check_generated_reply(reply,self.context)

    def test_graph_has_separate_evidence_and_model_nodes(self):
        graph=create_workflow()
        nodes=set(graph.nodes)
        self.assertTrue({'gather','reduce','generate','finalize'}.issubset(nodes))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    args,rest=parser.parse_known_args();DATA=args.data_dir.resolve()
    try: import langgraph
    except ImportError:parser.exit(2,'Workflow tests require requirements-assistant.txt.\n')
    unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
