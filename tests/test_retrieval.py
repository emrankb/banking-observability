#!/usr/bin/env python3
"""Exercise the real FAISS index, saved files, filters, and command-line search."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.knowledge_retrieval import KnowledgeStore, build_index, document_text, require_faiss
from banking_observability.forecasting import file_sha256

DATA = ROOT / 'data/raw'


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='banking-retrieval-test-')
        cls.root = Path(cls.temp.name)
        cls.index_dir = cls.root/'index'
        cls.manifest = build_index(DATA,cls.index_dir)
        cls.store = KnowledgeStore(cls.index_dir)
        cls.vectors = cls.store.encoder.transform([document_text(c) for c in cls.store.chunks]).toarray().astype('float32')
        cls.original = {p.name:file_sha256(p) for p in DATA.iterdir() if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def altered_copy(self, name):
        path=self.root/name;shutil.copytree(self.index_dir,path);return path

    def test_native_index_is_exact_inner_product(self):
        faiss=require_faiss()
        self.assertIsInstance(self.store.index,faiss.IndexFlatIP)
        self.assertEqual(self.store.index.ntotal,len(self.store.chunks))
        reconstructed=self.store.index.reconstruct_n(0,self.store.index.ntotal)
        np.testing.assert_allclose(reconstructed,self.vectors,atol=1e-7)

    def test_native_search_scores_match_independent_cosine(self):
        query=self.store.encoder.transform(['NEFT transaction gateway timeout']).toarray().astype('float32')
        scores,indices=self.store.index.search(query,len(self.store.chunks))
        expected=self.vectors@query[0]
        np.testing.assert_allclose(scores[0],expected[indices[0]],atol=1e-6)

    def test_reloaded_store_returns_same_results(self):
        query='database connection pool exhausted'
        self.assertEqual(self.store.search(query),KnowledgeStore(self.index_dir).search(query))

    def test_runbook_search_has_sources_and_relevant_first_hit(self):
        result=self.store.search('NEFT transaction gateway timeout')
        self.assertEqual(result['status'],'matches')
        self.assertEqual(result['results'][0]['document_id'],'runbook:neft_gateway_timeout')
        self.assertTrue(result['results'][0]['references'])
        self.assertEqual(result['method'],'tfidf_faiss_cosine')
        self.assertNotIn('answer',result)

    def test_filter_is_applied_before_top_k_selection(self):
        reply=self.store.search('CPU utilization',source_kind='supplied_runbook',top_k=3,min_similarity=0.01)
        self.assertTrue(reply['results'])
        self.assertTrue(all(row['kind']=='supplied_runbook' for row in reply['results']))
        query=self.store.encoder.transform(['CPU utilization']).toarray()[0]
        scores=self.vectors@query
        candidates=sorted([(float(score),c) for score,c in zip(scores,self.store.chunks)
                           if c['kind']=='supplied_runbook' and score >= 0.01],key=lambda r:(-round(r[0],7),r[1]['chunk_id']))
        expected=[];seen=set()
        for _,c in candidates:
            if c['document_id'] not in seen: expected.append(c['chunk_id']);seen.add(c['document_id'])
        self.assertEqual([r['chunk_id'] for r in reply['results']],expected[:3])

    def test_one_passage_per_document(self):
        reply=self.store.search('network gateway timeout transaction',source_kind='all',top_k=20,min_similarity=.01)
        docs=[r['document_id'] for r in reply['results']]
        self.assertEqual(len(docs),len(set(docs)))

    def test_default_search_excludes_incident_snapshots(self):
        reply=self.store.search('CPU memory Critical',min_similarity=.01)
        self.assertTrue(all(r['kind']!='incident_snapshot' for r in reply['results']))

    def test_incident_filter_retains_audit_and_source_rows(self):
        reply=self.store.search('INC000012',source_kind='incident_snapshot',min_similarity=.01)
        first=reply['results'][0]
        self.assertEqual(first['document_id'],'incident:INC000012')
        self.assertEqual(len(first['monitoring_disagreements']),5)
        self.assertEqual(len(first['references']),2)
        self.assertFalse(first['text_is_verbatim'])

    def test_out_of_vocabulary_query_returns_no_results(self):
        reply=self.store.search('telescope orbital galaxy')
        self.assertEqual(reply['status'],'no_vocabulary_overlap')
        self.assertEqual(reply['results'],[])

    def test_high_threshold_does_not_force_results(self):
        reply=self.store.search('CPU',min_similarity=1.0)
        self.assertEqual(reply['results'],[])

    def test_query_and_filter_validation(self):
        for query in ['', ' ', 'x'*4097]:
            with self.subTest(query=query[:10]), self.assertRaises(ValueError): self.store.search(query)
        for value in [0,21,True,1.2]:
            with self.subTest(k=value), self.assertRaises(ValueError): self.store.search('CPU',top_k=value)
        for value in [0,-1,1.1,float('nan')]:
            with self.subTest(minimum=value), self.assertRaises(ValueError): self.store.search('CPU',min_similarity=value)
        with self.assertRaises(ValueError): self.store.search('CPU',source_kind='missing')

    def test_replay_before_index_cutoff_is_rejected(self):
        for value in ['2025-10-20 00:00:00','2025-10-26T00:00:00Z','NaT']:
            with self.subTest(value=value), self.assertRaises(ValueError): self.store.search('CPU',as_of=value)
        reply=self.store.search('CPU',as_of='2025-10-25 18:01:00')
        self.assertEqual(reply['as_of'],'2025-10-25T18:01:00')

    def test_get_document_expands_sections_without_mutating_store(self):
        doc=self.store.get_document('runbook:high_cpu')
        self.assertEqual(len(doc),2)
        doc[0]['text']='changed'
        self.assertNotEqual(self.store.get_document('runbook:high_cpu')[0]['text'],'changed')
        with self.assertRaises(KeyError): self.store.get_document('missing')

    def test_no_later_incident_is_indexed(self):
        self.assertNotIn('incident:INC000136',{c['document_id'] for c in self.store.chunks})
        self.assertEqual(self.manifest['sources']['excluded_later_incident_rows'],33)

    def test_changed_source_is_reported(self):
        temp=self.root/'changed_sources';shutil.copytree(DATA,temp)
        self.assertEqual(self.store.source_changes(temp),[])
        p=temp/'incident_knowledge_base.txt';p.write_text(p.read_text()+'\n')
        self.assertEqual(self.store.source_changes(temp),['data/raw/incident_knowledge_base.txt'])

    def test_corrupt_native_index_is_refused(self):
        path=self.altered_copy('corrupt_native')
        with (path/'passages.faiss').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):KnowledgeStore(path)

    def test_corrupt_encoder_is_refused(self):
        path=self.altered_copy('corrupt_encoder');(path/'encoder.joblib').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):KnowledgeStore(path)

    def test_corrupt_passages_are_refused(self):
        path=self.altered_copy('corrupt_passages');p=path/'passages.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):KnowledgeStore(path)

    def test_manifest_version_and_shape_are_checked(self):
        for name,field,value in [('bad_version','format_version',999),('bad_shape','rows',0)]:
            path=self.altered_copy(name);p=path/'manifest.json';m=json.loads(p.read_text());m[field]=value;p.write_text(json.dumps(m))
            with self.assertRaises(ValueError):KnowledgeStore(path)

    def test_runtime_and_code_mismatch_are_refused(self):
        for field in ['runtime_versions','code_sha256']:
            path=self.altered_copy(field);p=path/'manifest.json';m=json.loads(p.read_text());m[field]={};p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'code/runtime'):KnowledgeStore(path)

    def test_existing_index_is_not_overwritten(self):
        hashes={p.name:file_sha256(p) for p in self.index_dir.iterdir()}
        with self.assertRaises(ValueError):build_index(DATA,self.index_dir)
        self.assertEqual(hashes,{p.name:file_sha256(p) for p in self.index_dir.iterdir()})

    def test_source_overlap_is_rejected(self):
        with self.assertRaises(ValueError):build_index(DATA,DATA/'index')

    def test_rebuild_reproduces_passages_vectors_and_search(self):
        path=self.root/'rebuilt';build_index(DATA,path)
        self.assertEqual(KnowledgeStore(path).chunks,self.store.chunks)
        self.assertEqual(KnowledgeStore(path).search('IMPS invalid account'),self.store.search('IMPS invalid account'))
        for name in ['passages.jsonl','passages.faiss']:
            self.assertEqual(file_sha256(path/name),file_sha256(self.index_dir/name))

    def test_build_and_search_clis(self):
        path=self.root/'cli'
        run=subprocess.run([sys.executable,str(ROOT/'scripts/build_knowledge.py'),'--data-dir',str(DATA),'--index-dir',str(path)],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        run=subprocess.run([sys.executable,str(ROOT/'scripts/search_knowledge.py'),'--index-dir',str(path),'--query','NEFT gateway timeout'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(json.loads(run.stdout)['results'][0]['document_id'],'runbook:neft_gateway_timeout')

    def test_authored_query_checks_and_report(self):
        out=self.root/'evaluation'
        run=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_retrieval.py'),'--index-dir',str(self.index_dir),
                            '--data-dir',str(DATA),'--output-dir',str(out)],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        result=json.loads((out/'query_results.json').read_text())
        positives=[r for r in result['rows'] if r['expected_document'] is not None]
        self.assertEqual(len(positives),13)
        self.assertEqual(result['hit_rate_at_k'],sum(r['rank'] is not None for r in positives)/len(positives))
        self.assertTrue(all(r['passed'] for r in result['rows']), result['rows'])
        self.assertIn('not an independent',result['scope'])

    def test_original_sources_unchanged(self):
        self.assertEqual(self.original,{p.name:file_sha256(p) for p in DATA.iterdir() if p.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-dir',type=Path,default=DATA)
    args,rest=parser.parse_known_args();DATA=args.data_dir.resolve()
    try: require_faiss()
    except (RuntimeError,ValueError) as exc: parser.exit(2,f'Retrieval tests not run: {exc}\n')
    unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
