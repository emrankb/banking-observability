#!/usr/bin/env python3
"""Check log grouping with original records and isolated input-error cases."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.forecasting import file_sha256
from banking_observability.log_clustering import (
    KMEANS_PARAMS, NORMALIZER_VERSION, SEED_OFFSETS, VECTOR_PARAMS, LogClassifier,
    lexical_text, message_evidence, normalize_message, read_catalog, read_logs, template_seed,
)
from banking_observability.log_experiment import assign_logs, duplicate_sensitivity

DATA_DIR: Path


def cli(name: str, *args, timeout=120):
    return subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


class LogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_hashes = {p.name: file_sha256(p) for p in DATA_DIR.glob('*.csv')}
        cls.temp = tempfile.TemporaryDirectory(prefix='log-tests-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.models = cls.directory / 'models'; cls.reports = cls.directory / 'reports'
        run = cli('train_logs.py', '--data-dir', DATA_DIR, '--model-dir', cls.models, '--report-dir', cls.reports)
        if run.returncode or run.stderr:
            raise RuntimeError(f'Training failed or warned:\n{run.stdout}\n{run.stderr}')
        cls.stdout = run.stdout
        cls.classifier = LogClassifier(cls.models); cls.model = cls.classifier.model
        cls.frame = read_logs(DATA_DIR / 'application_logs.csv')
        cls.cutoff = pd.Timestamp(pd.read_csv(DATA_DIR / 'infrastructure_metrics_train.csv').timestamp.iloc[-1])
        cls.dev = cls.frame.loc[cls.frame.timestamp.le(cls.cutoff)]
        cls.evaluation = cls.frame.loc[cls.frame.timestamp.gt(cls.cutoff)]
        cls.summary = json.loads((cls.reports / 'summary.json').read_text())
        cls.plan = json.loads((cls.reports / 'selection.json').read_text())
        cls.selection = pd.read_csv(cls.reports / 'cluster_selection.csv')
        cls.assignments = pd.read_csv(cls.reports / 'log_assignments.csv', keep_default_na=False)

    def test_snapshot_and_temporal_split(self):
        self.assertEqual(len(self.frame), 5000)
        self.assertEqual((len(self.dev), len(self.evaluation)), (3967, 1033))
        self.assertEqual(len(self.model.templates), 17)
        self.assertEqual(self.plan['training_end'], str(self.dev.timestamp.max()))
        self.assertLessEqual(self.dev.timestamp.max(), self.cutoff)
        self.assertGreater(self.evaluation.timestamp.min(), self.cutoff)
        self.assertEqual(self.summary['split']['evaluation_unseen_rows'], 0)

    def test_dynamic_ids_percentages_and_severity_are_not_features(self):
        row = self.frame.loc[self.frame.log_message.str.contains('High CPU')].iloc[0]
        changed = re.sub(r'srv-\d+', 'srv-999', row.log_message)
        changed = re.sub(r'Current: \d+%', 'Current: 77.25%', changed).replace('ERROR:', 'WARNING:')
        self.assertEqual(normalize_message(row.log_message), normalize_message(changed))
        terms = self.model.vectorizer.get_feature_names_out()
        self.assertFalse(any(re.search(r'\d|<|>|\b(error|info|warning|critical|txn|usr|srv)\b', t) for t in terms))

    def test_timestamp_prefix_is_masked(self):
        row = self.frame.iloc[1]
        wrapped = f'[2025-10-20 00:01:00] {row.log_message}'
        self.assertEqual(normalize_message(wrapped), normalize_message(row.log_message))

    def test_normalization_is_idempotent_for_source_templates(self):
        for template in self.model.templates:
            self.assertEqual(normalize_message(template), template)

    def test_failure_reasons_and_success_wording_are_preserved(self):
        names = ['neft', 'imps', 'rtgs']
        found = [next(t for t in self.model.templates if t.startswith(n)) for n in names]
        self.assertIn('gateway timeout', found[0]); self.assertIn('invalid account', found[1])
        self.assertIn('insufficient funds', found[2]); self.assertEqual(len(set(found)), 3)
        self.assertTrue(any('processed successfully' in t for t in self.model.templates))

    def test_fit_uses_unique_development_templates(self):
        self.assertEqual(self.model.templates, sorted(self.dev.normalized_message.unique()))
        self.assertEqual(len(self.model.estimator.labels_), 17)
        X = self.model.vectorizer.transform(self.model.templates)
        document_counts = np.asarray((X > 0).sum(axis=0)).ravel()
        expected_idf = np.log((1 + len(self.model.templates)) / (1 + document_counts)) + 1
        np.testing.assert_allclose(self.model.vectorizer.idf_, expected_idf)
        self.assertEqual(self.model.estimator.n_init, 20)

    def test_k_search_and_selection_recompute(self):
        self.assertEqual(self.plan['candidate_k'], list(range(2, 17)))
        details = pd.read_csv(self.reports / 'cluster_seed_results.csv')
        self.assertEqual(len(details), 15 * 3)
        means = details.groupby('k').silhouette.mean()
        for row in self.selection.itertuples(index=False):
            self.assertAlmostEqual(row.mean_template_silhouette, means.loc[row.k], places=12)
        chosen = self.selection.sort_values(['mean_template_silhouette', 'mean_seed_ari', 'k'], ascending=[False, False, True]).iloc[0]
        self.assertEqual(int(chosen.k), self.plan['selected_k'])

    def test_final_silhouette_recomputes(self):
        X = self.model.vectorizer.transform(self.model.templates)
        expected = silhouette_score(X, self.model.estimator.labels_, metric='cosine')
        self.assertAlmostEqual(self.plan['final_template_silhouette'], expected, places=12)

    def test_seed_uses_only_training_patterns(self):
        self.assertEqual(self.plan['seed'], template_seed(sorted(self.dev.normalized_message.unique())))
        self.assertEqual(self.plan['seed_offsets'], list(SEED_OFFSETS))

    def test_all_source_rows_and_duplicate_records_are_retained(self):
        self.assertEqual(len(self.assignments), len(self.frame))
        self.assertEqual(self.assignments.csv_row.tolist(), list(range(2, 5002)))
        self.assertEqual(self.assignments.log_message.tolist(), self.frame.log_message.tolist())
        self.assertEqual(int(self.assignments.duplicate_after_first.sum()), 9)

    def test_original_values_are_flagged_without_clipping(self):
        q = self.summary['quality']
        self.assertEqual(q, {'duplicate_rows': 9, 'severity_mismatches': 294, 'cpu_outside_0_100': 281, 'memory_outside_0_100': 286})
        self.assertEqual(pd.to_numeric(self.assignments.cpu_pct, errors='coerce').max(), 996)
        self.assertEqual(pd.to_numeric(self.assignments.memory_pct, errors='coerce').max(), 996)

    def test_severity_mismatch_is_preserved_in_inference(self):
        row = self.frame.loc[self.frame.log_message.str.startswith('CRITICAL:')].iloc[0]
        response = self.classifier.predict(row.log_message, row.log_level)
        self.assertEqual(response['message_level'], 'CRITICAL')
        self.assertIn('structured_message_severity_mismatch', response['quality_flags'])
        self.assertTrue(response['label_review_required'])

    def test_catalog_labels_are_explicit_and_not_cluster_predictions(self):
        catalog = read_catalog(ROOT / 'config/log_patterns.json')
        for row in self.frame.drop_duplicates('normalized_message').itertuples(index=False):
            answer = self.classifier.predict(row.log_message, row.log_level)
            self.assertEqual(answer['category_source'], 'exact_normalized_template_catalog')
            self.assertEqual(answer['category'], catalog[row.normalized_message]['category'])
            self.assertEqual(answer['event_type'], catalog[row.normalized_message]['event_type'])
        self.assertEqual(self.summary['category_counts'], {'business_kpi': 1736, 'infrastructure': 2025, 'routine': 1239})

    def test_known_template_without_catalog_label_remains_unreviewed(self):
        model = deepcopy(self.model); row = self.frame.iloc[1]
        del model.catalog[row.normalized_message]
        answer = model.classify(row.log_message, row.log_level)
        self.assertEqual(answer['category'], 'unreviewed')
        self.assertTrue(answer['label_review_required'])

    def test_mixed_groups_are_not_given_one_issue_category(self):
        mixed = [p for p in self.model.profiles if len(p['categories']) > 1]
        self.assertEqual(len(mixed), self.summary['mixed_category_clusters'])
        self.assertGreater(len(mixed), 0)
        for profile in mixed:
            self.assertTrue(profile['mixed_categories'])
            self.assertEqual(profile['label'], ' / '.join(profile['top_terms'][:3]))

    def test_profile_examples_are_real_development_records(self):
        profiles = json.loads((self.reports / 'cluster_profiles.json').read_text())
        count = 0
        for p in profiles:
            for example in p['examples']:
                source = self.frame.loc[self.frame.csv_row.eq(example['csv_row'])].iloc[0]
                self.assertLessEqual(source.timestamp, self.cutoff)
                self.assertEqual(example['message'], source.log_message)
                self.assertIn(source.normalized_message, p['templates']); count += 1
        self.assertEqual(count, 17)

    def test_duplicate_sensitivity_recomputes(self):
        expected = duplicate_sensitivity(self.model, self.dev, self.plan['seed'])
        self.assertEqual(expected, self.summary['duplicate_sensitivity'])
        self.assertGreater(expected['row_sample_cluster_silhouette'], self.plan['final_template_silhouette'])
        self.assertAlmostEqual(expected['exact_template_partition_row_silhouette'], 1.0, places=10)

    def test_unseen_wording_is_not_given_a_known_issue_type(self):
        row = self.frame.loc[self.frame.log_message.str.contains('Invalid account')].iloc[0]
        changed = row.log_message.replace('Invalid account', 'Account validation unavailable')
        answer = self.classifier.predict(changed, row.log_level)
        self.assertFalse(answer['template_seen_during_fit'])
        self.assertEqual(answer['category'], 'unreviewed'); self.assertIsNone(answer['event_type'])
        self.assertTrue(answer['label_review_required']); self.assertIsNotNone(answer['cluster_id'])

    def test_zero_vocabulary_does_not_force_a_cluster(self):
        answer = self.classifier.predict('qzxvplk', 'ERROR')
        self.assertIsNone(answer['cluster_id']); self.assertIsNone(answer['nearest_training_template'])
        self.assertEqual(answer['category'], 'unreviewed'); self.assertTrue(answer['label_review_required'])

    def test_empty_and_placeholder_only_inputs_are_rejected(self):
        for value in ('', '  ', None, 123, 'ERROR: 12345'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.classifier.predict(value, 'ERROR')
        with self.assertRaises(ValueError):
            self.classifier.predict(self.frame.iloc[0].log_message, '')

    def test_saved_inference_matches_assignments(self):
        sample = self.frame.drop_duplicates('normalized_message')
        for row in sample.itertuples(index=False):
            response = self.classifier.predict(row.log_message, row.log_level)
            recorded = self.assignments.loc[self.assignments.csv_row.eq(row.csv_row)].iloc[0]
            self.assertEqual(response['cluster_id'], recorded.cluster_id)
            self.assertEqual(response['category'], recorded.category)
            self.assertAlmostEqual(response['nearest_training_template']['cosine_similarity'], 1.0, places=10)

    def test_csv_cli_loads_saved_artifact(self):
        run = cli('classify_log.py', '--model-dir', self.models, '--log-file', DATA_DIR / 'application_logs.csv', '--csv-row', 3)
        self.assertEqual(run.returncode, 0, run.stderr); self.assertEqual(run.stderr, '')
        answer = json.loads(run.stdout)
        self.assertEqual(answer['event_type'], 'neft_gateway_timeout')
        self.assertEqual(answer['model_sha256'], self.classifier.metadata['model_sha256'])

    def test_message_cli_and_bad_options(self):
        row = self.frame.iloc[1]
        run = cli('classify_log.py', '--model-dir', self.models, '--message', row.log_message, '--log-level', row.log_level)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['event_type'], 'neft_gateway_timeout')
        for args in [('--csv-row', 1), ('--csv-row', 5002), ('--message', row.log_message), ('--csv-row', 3, '--log-level', 'INFO')]:
            bad = cli('classify_log.py', '--model-dir', self.models, '--log-file', DATA_DIR / 'application_logs.csv', *args)
            self.assertNotEqual(bad.returncode, 0); self.assertIn('Log classification failed:', bad.stderr)

    def test_source_hashes_and_outputs_are_consistent(self):
        self.assertEqual(self.raw_hashes, {p.name: file_sha256(p) for p in DATA_DIR.glob('*.csv')})
        for name, sha in self.summary['source_sha256'].items():
            self.assertEqual(sha, file_sha256(DATA_DIR / name))
        self.assertEqual(self.classifier.metadata['source_sha256'], self.summary['source_sha256'])
        self.assertEqual(self.classifier.metadata['catalog_sha256'], file_sha256(ROOT / 'config/log_patterns.json'))

    def test_output_paths_and_overwrite_protection(self):
        run = cli('train_logs.py', '--data-dir', DATA_DIR, '--model-dir', self.models, '--report-dir', self.reports)
        self.assertNotEqual(run.returncode, 0); self.assertIn('already exists', run.stderr)
        run = cli('train_logs.py', '--data-dir', DATA_DIR, '--model-dir', DATA_DIR / 'models', '--report-dir', self.directory / 'other')
        self.assertNotEqual(run.returncode, 0); self.assertIn('separate from input', run.stderr)
        run = cli('train_logs.py', '--data-dir', DATA_DIR, '--model-dir', self.directory / 'same', '--report-dir', self.directory / 'same')
        self.assertNotEqual(run.returncode, 0); self.assertIn('non-nested', run.stderr)

    def test_missing_inputs_fail_without_output_directories(self):
        missing = self.directory / 'missing'; dest = self.directory / 'no-input-model'
        run = cli('train_logs.py', '--data-dir', missing, '--model-dir', dest, '--report-dir', self.directory / 'no-input-report')
        self.assertNotEqual(run.returncode, 0); self.assertIn('Missing input:', run.stderr)
        self.assertFalse(dest.exists())

    def test_log_schema_missing_values_and_order_rejected(self):
        original = pd.read_csv(DATA_DIR / 'application_logs.csv').head(8)
        cases = [original.drop(columns='log_message'), original.iloc[::-1], original.iloc[:0]]
        bad = original.copy(); bad.loc[0, 'log_message'] = ''; cases.append(bad)
        bad = original.copy(); bad.loc[0, 'timestamp'] = 'not-a-time'; cases.append(bad)
        bad = original.copy(); bad['timestamp'] = pd.to_datetime(bad.timestamp).dt.tz_localize('UTC'); cases.append(bad)
        for i, frame in enumerate(cases):
            path = self.directory / f'invalid-{i}.csv'; frame.to_csv(path, index=False)
            with self.subTest(case=i), self.assertRaises(ValueError):
                read_logs(path)

    def test_catalog_rejects_invalid_and_duplicate_entries(self):
        catalog = json.loads((ROOT / 'config/log_patterns.json').read_text())
        variants = []
        bad = deepcopy(catalog); bad['patterns'].append(bad['patterns'][0]); variants.append(bad)
        bad = deepcopy(catalog); bad['patterns'][0]['category'] = 'made-up'; variants.append(bad)
        bad = deepcopy(catalog); bad['format_version'] = 99; variants.append(bad)
        for i, content in enumerate(variants):
            path = self.directory / f'catalog-{i}.json'; path.write_text(json.dumps(content))
            with self.subTest(case=i), self.assertRaises(ValueError): read_catalog(path)

    def test_checksum_and_metadata_mismatches_are_rejected(self):
        for kind in ['checksum', 'runtime', 'kind', 'code', 'normalizer']:
            dest = self.directory / f'bad-{kind}'; shutil.copytree(self.models, dest)
            meta = json.loads((dest / 'metadata.json').read_text())
            if kind == 'checksum': (dest / 'model.joblib').write_bytes(b'broken artifact')
            elif kind == 'runtime': meta['runtime_versions']['numpy'] = '0.0.0'
            elif kind == 'kind': meta['kind'] = 'incident_triage'
            elif kind == 'code': meta['code_sha256']['log_clustering.py'] = '0' * 64
            else: meta['normalizer_version'] = 99
            (dest / 'metadata.json').write_text(json.dumps(meta))
            with self.subTest(kind=kind), self.assertRaises(ValueError): LogClassifier(dest)

    def test_later_wording_does_not_change_fitting(self):
        data = self.directory / 'changed-later'; data.mkdir()
        source = pd.read_csv(DATA_DIR / 'application_logs.csv')
        mask = pd.to_datetime(source.timestamp) > self.cutoff
        # Change later messages to check that fitting uses only the earlier records.
        source.loc[mask, 'log_message'] += ' unseenword'
        source.to_csv(data / 'application_logs.csv', index=False)
        shutil.copyfile(DATA_DIR / 'infrastructure_metrics_train.csv', data / 'infrastructure_metrics_train.csv')
        models, reports = self.directory / 'later-model', self.directory / 'later-report'
        run = cli('train_logs.py', '--data-dir', data, '--model-dir', models, '--report-dir', reports)
        self.assertEqual(run.returncode, 0, run.stderr)
        altered = LogClassifier(models).model
        self.assertEqual(altered.templates, self.model.templates)
        np.testing.assert_array_equal(altered.vectorizer.idf_, self.model.vectorizer.idf_)
        np.testing.assert_array_equal(altered.estimator.cluster_centers_, self.model.estimator.cluster_centers_)
        self.assertEqual(json.loads((reports / 'selection.json').read_text()), self.plan)
        later = pd.read_csv(reports / 'log_assignments.csv')
        self.assertTrue(later.loc[later.split.eq('evaluation'), 'category'].eq('unreviewed').all())

    def test_full_rerun_reproduces_tables_and_selection(self):
        models, reports = self.directory / 'again-model', self.directory / 'again-report'
        run = cli('train_logs.py', '--data-dir', DATA_DIR, '--model-dir', models, '--report-dir', reports)
        self.assertEqual(run.returncode, 0, run.stderr); self.assertEqual(run.stderr, '')
        for original in self.reports.iterdir():
            self.assertEqual(original.read_bytes(), (reports / original.name).read_bytes(), original.name)

    def test_notebook_source_is_valid_and_outputs_are_empty(self):
        book = nbformat.read(ROOT / 'notebooks/04_log_clustering.ipynb', as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertIsNone(cell.execution_count); self.assertEqual(cell.outputs, [])

    def test_report_names_every_fitted_pattern_without_claiming_accuracy(self):
        text = (self.reports / 'log_review.md').read_text()
        self.assertIn('LOG CLUSTERS:', self.stdout)
        self.assertIn('not a test of new failure modes', text)
        self.assertIn('Category tags come from the explicit template catalog', text)
        templates = pd.read_csv(self.reports / 'template_assignments.csv')
        self.assertEqual(sorted(templates.normalized_message), self.model.templates)
        self.assertEqual(int(templates.development_rows.sum()), len(self.dev))
        self.assertEqual(int(templates.evaluation_rows.sum()), len(self.evaluation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw')
    args = parser.parse_args(); DATA_DIR = args.data_dir.resolve()
    unittest.main(argv=[sys.argv[0]], verbosity=2)
