#!/usr/bin/env python3
"""Check real source passages and lexical encoding without a FAISS dependency."""
from pathlib import Path
import argparse
import copy
from hashlib import sha256
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.knowledge_sources import (collect_knowledge, csv_spans, line_windows, MAX_PASSAGE_CHARS,
    naive_time, preview_sources, runbook_chunks, source_ref, validate_chunk)
from banking_observability.knowledge_retrieval import document_text, fit_encoder
from banking_observability.forecasting import file_sha256

DATA = ROOT / "data/raw"


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.summary = collect_knowledge(DATA)
        cls.encoder, cls.vectors = fit_encoder(cls.chunks)
        cls.before = {p.name: file_sha256(p) for p in DATA.iterdir() if p.is_file()}

    def test_counts_recompute_from_sources(self):
        raw = pd.read_csv(DATA / 'incident_reports.csv')
        selected = pd.to_datetime(raw.timestamp) <= pd.Timestamp(self.summary['cutoff'])
        self.assertEqual(self.summary['included_incident_rows'], int(selected.sum()))
        self.assertEqual(self.summary['excluded_later_incident_rows'], int((~selected).sum()))
        self.assertEqual(self.summary['documents_by_kind'], {'supplied_runbook': 5, 'project_runbook': 13, 'incident_snapshot': 135})
        self.assertEqual(len(self.chunks), self.summary['chunk_count'])

    def test_all_citation_spans_match_real_source(self):
        for chunk in self.chunks:
            validate_chunk(chunk)
            for ref in chunk['references']:
                path = DATA / Path(ref['path']).name if ref['path'].startswith('data/raw/') else ROOT / ref['path']
                self.assertEqual(source_ref(path, ref['path'], ref['line_start'], ref['line_end']), ref)

    def test_supplied_passages_are_unchanged(self):
        chunks = [c for c in self.chunks if c['kind'] == 'supplied_runbook']
        self.assertEqual(len({c['document_id'] for c in chunks}), 5)
        for c in chunks:
            self.assertTrue(c['text_is_verbatim'])
            self.assertEqual(c['text'], c['references'][0]['excerpt'])

    def test_authored_notes_have_separate_provenance(self):
        for c in self.chunks:
            if c['kind'] == 'project_runbook':
                self.assertEqual(c['references'][0]['path'], 'knowledge/runbooks.md')
                self.assertIsNone(c['event_timestamp'])

    def test_every_nonroutine_catalog_event_has_notes(self):
        catalog = json.loads((ROOT/'config/log_patterns.json').read_text())
        expected = {r['event_type'] for r in catalog['patterns'] if r['category'] != 'routine'}
        self.assertEqual(set(self.summary['runbook_event_types']), expected)

    def test_incident_cutoff_excludes_later_snapshots(self):
        times = [pd.Timestamp(c['event_timestamp']) for c in self.chunks if c['event_timestamp']]
        self.assertTrue(all(t <= pd.Timestamp(self.summary['cutoff']) for t in times))
        self.assertNotIn('incident:INC000136', [c['document_id'] for c in self.chunks])

    def test_earlier_cutoff_rebuilds_the_corpus(self):
        earlier, summary = collect_knowledge(DATA, cutoff='2025-10-22 12:00:00')
        self.assertLess(summary['included_incident_rows'], self.summary['included_incident_rows'])
        self.assertTrue(all(not c['event_timestamp'] or pd.Timestamp(c['event_timestamp']) <= pd.Timestamp(summary['cutoff']) for c in earlier))

    def test_cutoff_must_be_an_observed_training_minute(self):
        for value in ['2025-10-27 00:00:00', '2025-10-20 00:00:30', '2025-10-20T00:00:00Z', 'NaT']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                collect_knowledge(DATA, cutoff=value)

    def test_incident_conflicts_recompute(self):
        raw = pd.read_csv(DATA/'incident_reports.csv').set_index('incident_id')
        monitoring = pd.read_csv(DATA/'infrastructure_metrics_train.csv').set_index('timestamp')
        count = 0
        for c in self.chunks:
            if c['kind'] != 'incident_snapshot': continue
            row = raw.loc[c['document_id'].split(':',1)[1]]
            for d in c['monitoring_disagreements']:
                self.assertEqual(d['incident_value'], float(row[d['metric']]))
                self.assertEqual(d['monitoring_value'], float(monitoring.loc[row.timestamp,d['metric']]))
                self.assertNotAlmostEqual(d['incident_value'],d['monitoring_value'])
                count += 1
        self.assertEqual(count,self.summary['included_incident_conflict_cells'])
        self.assertEqual(count,100)

    def test_conflicts_do_not_replace_incident_values(self):
        c = next(c for c in self.chunks if c['document_id'] == 'incident:INC000012')
        self.assertEqual(len(c['monitoring_disagreements']),5)
        self.assertEqual(len(c['references']),2)
        self.assertIn('87.65349821141189',c['references'][0]['excerpt'])
        self.assertIn('72.99',c['references'][1]['excerpt'])

    def test_snapshot_summaries_are_not_marked_verbatim(self):
        for c in self.chunks:
            if c['kind'] == 'incident_snapshot':
                self.assertFalse(c['text_is_verbatim'])
                self.assertIn('No confirmed root cause, resolution, or recovery outcome',c['text'])

    def test_record_ids_and_source_rows_are_exact(self):
        spans = csv_spans(DATA/'incident_reports.csv','incident_id')
        self.assertEqual(spans['INC000012'], (13,13))
        self.assertEqual(spans['INC000168'], (169,169))

    def test_passages_have_unique_stable_ids(self):
        repeated, summary = collect_knowledge(DATA)
        self.assertEqual(repeated,self.chunks)
        self.assertEqual(summary,self.summary)
        self.assertEqual(len(set(c['chunk_id'] for c in repeated)), len(repeated))

    def test_runbook_chunks_are_bounded(self):
        for c in self.chunks:
            if c['kind'] != 'incident_snapshot':
                self.assertLessEqual(len(c['text']),MAX_PASSAGE_CHARS)

    def test_long_sections_split_with_real_text(self):
        lines = (DATA/'incident_knowledge_base.txt').read_text().splitlines()*5
        spans = list(line_windows(lines,1,len(lines)))
        covered = set()
        for a,b in spans:
            covered.update(range(a,b+1))
            self.assertLessEqual(len('\n'.join(lines[a-1:b])), MAX_PASSAGE_CHARS)
        self.assertTrue(all(i in covered for i,line in enumerate(lines,1) if line.strip()))

    def test_overlong_source_line_is_rejected(self):
        source = (DATA/'incident_knowledge_base.txt').read_text().replace('\n',' ')
        with self.assertRaises(ValueError): list(line_windows([source],1,1))

    def test_duplicate_titles_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'notes.txt'
            text=(DATA/'incident_knowledge_base.txt').read_text();p.write_text(text+'\n'+text)
            with self.assertRaisesRegex(ValueError,'Duplicate document'): runbook_chunks(p,'duplicate.txt')

    def test_missing_input_is_actionable(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(FileNotFoundError,'incident_knowledge_base.txt'):
            collect_knowledge(Path(temp))

    def test_catalog_coverage_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'catalog.json';data=json.loads((ROOT/'config/log_patterns.json').read_text())
            data['patterns']=[row for row in data['patterns'] if row['event_type'] != 'high_cpu']
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Runbook/catalog mismatch'): collect_knowledge(DATA,catalog=path)

    def test_citation_and_text_tampering_are_rejected(self):
        c=copy.deepcopy(self.chunks[0]);c['references'][0]['excerpt']+=' changed'
        with self.assertRaisesRegex(ValueError,'checksum'): validate_chunk(c)
        c=copy.deepcopy(self.chunks[0]);c['text']+=' changed'
        with self.assertRaisesRegex(ValueError,'identity'): validate_chunk(c)

    def test_unsafe_citation_path_is_rejected(self):
        for name in ['/etc/passwd','../../private']:
            c=copy.deepcopy(self.chunks[0]);c['references'][0]['path']=name
            with self.assertRaisesRegex(ValueError,'citation path'): validate_chunk(c)

    def test_preview_cli_succeeds_without_faiss(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'report'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/inspect_knowledge.py'),'--data-dir',str(DATA),'--output-dir',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stderr,'')
            self.assertTrue((out/'source_review.md').is_file())
            self.assertEqual(json.loads((out/'summary.json').read_text()),self.summary)

    def test_preview_refuses_overwrite_and_data_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'report';preview_sources(DATA,out)
            with self.assertRaises(ValueError): preview_sources(DATA,out)
        with self.assertRaises(ValueError): preview_sources(DATA,DATA/'generated')

    def test_sparse_encoder_vectors_are_finite_and_normalized(self):
        self.assertEqual(self.vectors.dtype,np.float32)
        self.assertTrue(self.vectors.flags.c_contiguous)
        np.testing.assert_allclose(np.linalg.norm(self.vectors,axis=1),1,atol=1e-6)
        self.assertEqual(self.vectors.shape[0],len(self.chunks))

    def test_encoder_roundtrip_matches_fitted_vectors(self):
        np.testing.assert_allclose(self.encoder.transform([document_text(c) for c in self.chunks]).toarray(),self.vectors,atol=1e-7)

    def test_zero_overlap_query_has_zero_vector(self):
        v=self.encoder.transform(['telescope orbital galaxy']).toarray()
        self.assertEqual(np.count_nonzero(v),0)

    def test_notebook_source_is_valid_and_outputs_empty(self):
        import nbformat
        notebook=nbformat.read(ROOT/'notebooks/05_knowledge_retrieval.ipynb',as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                compile(cell.source,'retrieval-notebook','exec')
                self.assertIsNone(cell.execution_count);self.assertEqual(cell.outputs,[])

    def test_inputs_are_unchanged(self):
        self.assertEqual(self.before,{p.name:file_sha256(p) for p in DATA.iterdir() if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,default=DATA)
    args,rest=parser.parse_known_args();DATA=args.data_dir.resolve()
    unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
