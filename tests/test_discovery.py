#!/usr/bin/env python3
"""Run the discovery CLIs and regression checks against the original CSV files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_discovery import EXPECTED_COLUMNS, LATENCY, normalize_log  # noqa: E402

DATA_DIR: Path
INPUTS = list(EXPECTED_COLUMNS) + [
    'incident_reports.csv', 'infrastructure_metrics_train.csv', 'infrastructure_metrics_test.csv',
]


def digests(directory: Path) -> dict[str, str]:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in INPUTS}


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [name for name in INPUTS if not (DATA_DIR / name).is_file()]
        if missing:
            raise FileNotFoundError('Full snapshot tests require these original files: ' + ', '.join(missing))
        cls.before = digests(DATA_DIR)
        cls.temporary = tempfile.TemporaryDirectory(prefix='banking-discovery-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output = Path(cls.temporary.name) / 'analysis'
        cls.profiles_dir = Path(cls.temporary.name) / 'inspection'
        cls.inspect_run = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/inspect_data.py'), '--data-dir', str(DATA_DIR), '--output-dir', str(cls.profiles_dir)],
            capture_output=True, text=True, check=True, timeout=60,
        )
        cls.analysis_run = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/analyze_discovery.py'), '--data-dir', str(DATA_DIR), '--output-dir', str(cls.output)],
            capture_output=True, text=True, check=True, timeout=60,
        )
        cls.audit = json.loads((cls.output / 'audit.json').read_text())
        cls.profiles = json.loads((cls.output / 'schema.json').read_text())
        cls.infra = pd.read_csv(DATA_DIR / 'infrastructure_metrics.csv', parse_dates=['timestamp'])
        cls.txn = pd.read_csv(DATA_DIR / 'transactions_fraud_train.csv', parse_dates=['timestamp'])
        cls.logs = pd.read_csv(DATA_DIR / 'application_logs.csv', parse_dates=['timestamp'])

    def test_both_clis_complete_without_stderr(self) -> None:
        self.assertEqual(self.inspect_run.returncode, 0)
        self.assertEqual(self.analysis_run.returncode, 0)
        self.assertEqual(self.inspect_run.stderr, '')
        self.assertEqual(self.analysis_run.stderr, '')
        self.assertIn('SAME-MINUTE CORRELATIONS', self.analysis_run.stdout)

    def test_source_bytes_unchanged(self) -> None:
        self.assertEqual(self.before, digests(DATA_DIR))
        for name, digest in self.audit['source_sha256'].items():
            self.assertEqual(digest, self.before[name])

    def test_schema_snapshot_and_inferred_types(self) -> None:
        expected_sizes = {'transactions_fraud_train.csv': (4000, 10), 'infrastructure_metrics.csv': (10080, 13), 'application_logs.csv': (5000, 3)}
        for profile in self.profiles:
            self.assertEqual((profile['rows'], profile['columns']), expected_sizes[profile['file']])
            self.assertEqual(tuple(c['column'] for c in profile['schema']), EXPECTED_COLUMNS[profile['file']])
        standalone = json.loads((self.profiles_dir / 'schema.json').read_text())
        self.assertEqual(standalone, self.profiles)

    def test_no_missing_values_or_invalid_timestamps_in_snapshot(self) -> None:
        for profile in self.profiles:
            self.assertEqual(sum(c['missing'] for c in profile['schema']), 0)
            self.assertEqual(sum(c.get('blank_strings', 0) for c in profile['schema']), 0)
            time = next(c for c in profile['schema'] if c['column'] == 'timestamp')
            self.assertEqual(time['invalid_timestamps'], 0)
            self.assertEqual(time['timezone'], 'None')

    def test_minute_grid_has_no_gaps(self) -> None:
        expected = pd.date_range(self.infra.timestamp.min(), self.infra.timestamp.max(), freq='min')
        self.assertEqual(len(expected), 10080)
        self.assertTrue(expected.equals(pd.DatetimeIndex(self.infra.timestamp)))

    def test_no_operational_status_in_fraud_data(self) -> None:
        self.assertFalse({'status', 'success', 'is_failed', 'failure_reason'}.intersection(self.txn.columns))
        self.assertEqual(int(self.txn.is_fraud.sum()), 191)
        self.assertEqual(self.txn.transaction_id.nunique(), len(self.txn))

    def test_timestamp_alignment_is_many_to_one_and_nonfuture(self) -> None:
        aligned = pd.read_csv(self.output / 'aligned_transactions.csv', parse_dates=['transaction_timestamp', 'metric_minute'])
        self.assertEqual(len(aligned), 313)
        self.assertEqual(aligned.metric_minute.nunique(), 311)
        age = (aligned.transaction_timestamp - aligned.metric_minute).dt.total_seconds()
        self.assertTrue(age.between(0, 60, inclusive='left').all())
        self.assertEqual(aligned.transaction_id.nunique(), len(aligned))
        self.assertEqual(self.audit['alignment']['exact_timestamp_matches'], 0)
        self.assertEqual(self.audit['alignment']['outside_monitoring_window_rows'], 3687)

    def test_correlations_recompute_from_raw_metrics(self) -> None:
        table = pd.read_csv(self.output / 'correlations.csv')
        for record in table.loc[table.resolution.eq('1min')].to_dict('records'):
            target = record['target']
            if target.endswith('_non_success_pct'):
                column = target.removesuffix('_non_success_pct') + '_success_rate'
                actual = 100 - self.infra[column]
            else:
                actual = self.infra[target]
            self.assertAlmostEqual(record['pearson_r'], self.infra[record['feature']].corr(actual), places=12)
            self.assertAlmostEqual(record['spearman_rho'], self.infra[record['feature']].rank().corr(actual.rank()), places=12)
            self.assertEqual(record['n'], len(self.infra))
        self.assertAlmostEqual(self.infra.cpu_utilization.corr(self.infra[LATENCY]), 0.992230391986, places=5)

    def test_high_cpu_comparison_uses_observed_percentile(self) -> None:
        threshold = self.infra.cpu_utilization.quantile(.95)
        self.assertAlmostEqual(threshold, 73.1705, places=8)
        self.assertEqual(int(self.infra.cpu_utilization.gt(threshold).sum()), 504)
        row = next(r for r in self.audit['infrastructure']['high_cpu_comparison'] if r['target'] == LATENCY)
        direct = self.infra.loc[self.infra.cpu_utilization.gt(threshold), LATENCY].mean()
        self.assertAlmostEqual(row['high_cpu_mean'], direct, places=10)

    def test_log_normalization_masks_attached_units_and_ids(self) -> None:
        normalized = self.logs.log_message.map(normalize_log)
        self.assertEqual(normalized.nunique(), 17)
        self.assertFalse(normalized.str.contains(r'\b(?:TXN|USR)\d+\b|\bsrv-\d+\b|\d+(?:\.\d+)?ms|\d+%', regex=True).any())
        families = pd.read_csv(self.output / 'log_message_families.csv')
        self.assertEqual(int(families.rows.sum()), len(self.logs))

    def test_log_data_quality_counts(self) -> None:
        audit = self.audit['logs']
        self.assertEqual(audit['exact_duplicate_rows_after_first'], 9)
        self.assertEqual(audit['severity_mismatches'], 294)
        self.assertEqual(audit['percentages']['logged_cpu_pct']['above_100'], 281)
        self.assertEqual(audit['percentages']['logged_memory_pct']['above_100'], 286)
        self.assertEqual(audit['exact_logged_transaction_id_matches_to_train'], 0)
        self.assertEqual(audit['logs_outside_infra_timestamp_index'], 0)

    def test_incident_disagreements_are_explicit(self) -> None:
        audit = self.audit['supporting']['incidents']
        self.assertEqual(audit['critical_counts'], {'0': 142, '1': 26})
        self.assertEqual(audit['mismatched_rows'], 26)
        self.assertEqual(audit['mismatched_cells'], 130)
        self.assertEqual(audit['noncritical_mismatched_rows'], 0)
        self.assertEqual(audit['cpu_gt_85_rule_matches'], 168)
        self.assertFalse(audit['rule_is_model_evaluation'])
        differences = pd.read_csv(self.output / 'incident_metric_disagreements.csv')
        self.assertTrue(differences.is_critical.eq(1).all())
        self.assertFalse(np.isclose(differences.incident_value, differences.monitoring_value, rtol=0, atol=1e-9).any())

    def test_existing_split_is_chronological_and_exact(self) -> None:
        audit = self.audit['supporting']['infrastructure_split']
        self.assertEqual(audit['train_rows'], 8064)
        self.assertEqual(audit['test_rows'], 2016)
        self.assertEqual(audit['timestamp_overlap'], 0)
        self.assertTrue(audit['train_strictly_before_test'])
        self.assertTrue(audit['concatenation_exactly_equals_full'])

    def test_repeated_execution_reproduces_generated_artifacts(self) -> None:
        # Reports are deliberately untracked. Rebuild from the same source files
        # instead of requiring a saved output directory from another machine.
        repeated = Path(self.temporary.name) / 'repeated'
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/analyze_discovery.py'),
             '--data-dir', str(DATA_DIR), '--output-dir', str(repeated)],
            capture_output=True, text=True, check=True, timeout=60,
        )
        self.assertEqual(result.stderr, '')
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {path.name for path in repeated.iterdir()},
        )
        for path in self.output.iterdir():
            with self.subTest(file=path.name):
                self.assertEqual(path.read_bytes(), (repeated / path.name).read_bytes())

    def test_primary_files_do_not_require_supporting_inputs(self) -> None:
        primary_dir = Path(self.temporary.name) / 'primary-only'
        primary_dir.mkdir()
        for name in EXPECTED_COLUMNS:
            shutil.copyfile(DATA_DIR / name, primary_dir / name)
        output = Path(self.temporary.name) / 'primary-analysis'
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/analyze_discovery.py'),
             '--data-dir', str(primary_dir), '--output-dir', str(output)],
            capture_output=True, text=True, check=True, timeout=60,
        )
        self.assertEqual(completed.stderr, '')
        audit = json.loads((output / 'audit.json').read_text())
        self.assertEqual(audit['supporting'], {
            'incidents': {'status': 'not_supplied'},
            'infrastructure_split': {'status': 'not_supplied'},
        })
        for key in ('infrastructure', 'alignment', 'logs'):
            self.assertEqual(audit[key], self.audit[key])
        self.assertFalse((output / 'incident_metric_disagreements.csv').exists())
        for name in EXPECTED_COLUMNS:
            self.assertEqual(
                hashlib.sha256((primary_dir / name).read_bytes()).hexdigest(),
                self.before[name],
            )

    def test_missing_input_directory_exits_with_actionable_error(self) -> None:
        missing = Path(self.temporary.name) / 'not-provided'
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/analyze_discovery.py'), '--data-dir', str(missing), '--output-dir', str(self.output)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn('Missing input files:', completed.stderr)
        self.assertIn('transactions_fraud_train.csv', completed.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args, unittest_args = parser.parse_known_args()
    DATA_DIR = args.data_dir.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
