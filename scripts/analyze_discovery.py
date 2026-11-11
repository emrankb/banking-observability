#!/usr/bin/env python3
"""Audit relationships in the supplied banking data; no training or data repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspect_data import FILES, inspect

CHANNELS = ('neft', 'imps', 'rtgs')
LOAD_COLUMNS = (
    'cpu_utilization', 'memory_usage', 'db_connection_pool_usage',
    'network_throughput_mbps', 'transaction_volume_per_hour',
    'application_error_rate',
)
LATENCY = 'avg_transaction_processing_time_ms'
EXPECTED_COLUMNS = {
    'transactions_fraud_train.csv': (
        'transaction_id', 'customer_id', 'timestamp', 'transaction_type', 'amount',
        'merchant_category', 'device_type', 'location', 'processing_time_ms', 'is_fraud',
    ),
    'infrastructure_metrics.csv': (
        'timestamp', 'cpu_utilization', 'memory_usage', 'db_connection_pool_usage',
        'network_throughput_mbps', 'application_error_rate', 'neft_success_rate',
        'imps_success_rate', 'rtgs_success_rate', LATENCY, 'auth_success_rate',
        'transaction_volume_per_hour', 'service_uptime',
    ),
    'application_logs.csv': ('timestamp', 'log_level', 'log_message'),
}


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = EXPECTED_COLUMNS.get(path.name)
    if expected is not None and tuple(frame.columns) != expected:
        raise ValueError(f'{path.name}: schema changed; expected {expected}, got {tuple(frame.columns)}')
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='raise')
    if frame['timestamp'].isna().any():
        raise ValueError(f'{path.name}: missing timestamps cannot be aligned')
    return frame


def pair_summary(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Descriptive coefficients only: no independence-based significance claims."""
    paired = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()
    nonconstant = len(paired) > 1 and paired['x'].nunique() > 1 and paired['y'].nunique() > 1
    return {
        'n': len(paired),
        'pearson_r': float(paired['x'].corr(paired['y'])) if nonconstant else None,
        # Ranking explicitly avoids an otherwise unnecessary SciPy dependency.
        'spearman_rho': float(paired['x'].rank().corr(paired['y'].rank())) if nonconstant else None,
    }


def normalize_log(message: str) -> str:
    """Count message families, not semantic clusters. Preserve wording and units."""
    message = re.sub(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?', '<TIMESTAMP>', message)
    message = re.sub(r'\bTXN\d+\b', '<TXN_ID>', message)
    message = re.sub(r'\bUSR\d+\b', '<USER_ID>', message)
    message = re.sub(r'\bsrv-\d+\b', '<SERVER_ID>', message)
    # No trailing word boundary: a number directly followed by "ms" must also be masked.
    return re.sub(r'\d+(?:\.\d+)?', '<NUM>', message)


def audit_relationships(infra: pd.DataFrame, out: Path) -> dict[str, Any]:
    frame = infra.sort_values('timestamp').set_index('timestamp').copy()
    for channel in CHANNELS:
        frame[f'{channel}_non_success_pct'] = 100 - frame[f'{channel}_success_rate']
    targets = [f'{c}_non_success_pct' for c in CHANNELS] + [LATENCY, 'auth_success_rate']
    records = []
    for resolution, values in [('1min', frame), ('15min', frame.resample('15min').mean()), ('1h', frame.resample('1h').mean())]:
        for feature in LOAD_COLUMNS:
            for target in targets:
                records.append({'resolution': resolution, 'feature': feature, 'target': target, **pair_summary(values[feature], values[target])})
    correlations = pd.DataFrame(records)
    correlations.to_csv(out / 'correlations.csv', index=False)

    # A large contemporaneous correlation can partly reflect a common daily schedule.
    residual = frame - frame.groupby(frame.index.hour).transform('mean')
    changes = frame.diff()
    sensitivity = []
    for label, values in [('first_difference', changes), ('hour_of_day_demeaned', residual)]:
        for feature in LOAD_COLUMNS:
            for target in targets:
                sensitivity.append({'transformation': label, 'feature': feature, 'target': target, **pair_summary(values[feature], values[target])})
    pd.DataFrame(sensitivity).to_csv(out / 'correlation_sensitivity.csv', index=False)

    # Descriptive full-sample percentile, NOT an SLO or a fitted alert threshold.
    threshold = float(frame.cpu_utilization.quantile(.95))
    high = frame.cpu_utilization.gt(threshold)
    comparison = []
    for target in targets:
        high_mean = float(frame.loc[high, target].mean())
        remaining_mean = float(frame.loc[~high, target].mean())
        comparison.append({'target': target, 'cpu_p95_threshold': threshold,
                           'high_cpu_minutes': int(high.sum()), 'remaining_minutes': int((~high).sum()),
                           'high_cpu_mean': high_mean, 'remaining_mean': remaining_mean,
                           'difference': high_mean - remaining_mean})
    pd.DataFrame(comparison).to_csv(out / 'high_cpu_comparison.csv', index=False)
    hourly = frame.groupby(frame.index.hour)[list(LOAD_COLUMNS) + targets].mean()
    hourly.index.name = 'hour_in_source_clock'
    hourly.to_csv(out / 'hour_of_day_means.csv')
    work_hours = frame.index.hour >= 9
    work_hours &= frame.index.hour < 18
    regimes = []
    for label, mask in [('09:00-17:59', work_hours), ('other_hours', ~work_hours)]:
        regimes.append({'period': label, 'minutes': int(mask.sum()),
                        'mean_cpu': float(frame.loc[mask, 'cpu_utilization'].mean()),
                        'mean_processing_ms': float(frame.loc[mask, LATENCY].mean())})
    uptime = infra.loc[infra.service_uptime.ne(100)].copy()
    uptime['timestamp'] = uptime.timestamp.astype(str)
    percent_columns = ['cpu_utilization', 'memory_usage', 'db_connection_pool_usage',
                       'neft_success_rate', 'imps_success_rate', 'rtgs_success_rate',
                       'auth_success_rate', 'service_uptime']
    return {
        'minute_correlations': correlations.loc[correlations.resolution.eq('1min')].to_dict('records'),
        'sensitivity': sensitivity,
        'high_cpu_comparison': comparison,
        'time_of_day_regimes': regimes,
        'cpu_above_85_minutes': int(frame.cpu_utilization.gt(85).sum()),
        'percent_out_of_0_100_counts': {c: int((~infra[c].between(0, 100)).sum()) for c in percent_columns},
        'latency_exactly_100ms_minutes': int(frame[LATENCY].eq(100).sum()),
        'non_100_uptime_records': uptime.to_dict('records'),
        'non_success_definition': '100 - source success percentage; NOT fraud, failed counts, or a confirmed final-failure definition',
    }


def audit_alignment(txn: pd.DataFrame, infra: pd.DataFrame, out: Path) -> dict[str, Any]:
    start, end_exclusive = infra.timestamp.min(), infra.timestamp.max() + pd.Timedelta(minutes=1)
    overlap = txn.timestamp.ge(start) & txn.timestamp.lt(end_exclusive)
    candidates = txn.loc[overlap].rename(columns={'timestamp': 'transaction_timestamp'}).copy()
    candidates['metric_minute'] = candidates.transaction_timestamp.dt.floor('min')
    matched = candidates.merge(infra.rename(columns={'timestamp': 'metric_minute'}), on='metric_minute', how='inner', validate='many_to_one')
    age = (matched.transaction_timestamp - matched.metric_minute).dt.total_seconds()
    matched.to_csv(out / 'aligned_transactions.csv', index=False)
    correlations = []
    for feature in LOAD_COLUMNS:
        correlations.append({'feature': feature, 'target': 'processing_time_ms', **pair_summary(matched[feature], matched.processing_time_ms)})
    pd.DataFrame(correlations).to_csv(out / 'transaction_latency_correlations.csv', index=False)
    coverage = {
        'source_rows': len(txn), 'overlap_rows': int(overlap.sum()), 'aligned_rows': len(matched),
        'outside_monitoring_window_rows': int((~overlap).sum()),
        'overlap_rows_without_metric': len(candidates) - len(matched),
        'exact_timestamp_matches': int(txn.timestamp.isin(infra.timestamp).sum()),
        'aligned_distinct_minutes': int(matched.metric_minute.nunique()),
        'metric_age_seconds_min': float(age.min()), 'metric_age_seconds_max': float(age.max()),
        'aligned_channel_counts': matched.transaction_type.value_counts().to_dict(),
        'aligned_fraud_rows': int(matched.is_fraud.sum()),
        'cpu_vs_transaction_processing_time': pair_summary(matched.cpu_utilization, matched.processing_time_ms),
        'assumption': 'Treat all naive timestamps as the same source clock; floor transactions to the recorded minute. No timezone or measurement-window definition was supplied.',
        'no_failure_outcome': 'No status or success/failure field in transaction CSV; is_fraud is not an operational failure outcome.',
    }
    return coverage


def audit_logs(logs: pd.DataFrame, txn: pd.DataFrame, infra: pd.DataFrame, out: Path) -> dict[str, Any]:
    frame = logs.copy()
    frame['csv_row_number'] = np.arange(2, len(frame) + 2)
    frame['normalized_message'] = frame.log_message.map(normalize_log)
    frame['message_level'] = frame.log_message.str.extract(r'^(\w+):')[0]
    frame['exact_duplicate_after_first'] = frame.duplicated(subset=list(EXPECTED_COLUMNS['application_logs.csv']))
    frame['level_mismatch'] = frame.log_level.ne(frame.message_level)
    frame['logged_cpu_pct'] = pd.to_numeric(frame.log_message.str.extract(r'High CPU utilization.*Current: (\d+(?:\.\d+)?)%')[0])
    frame['logged_memory_pct'] = pd.to_numeric(frame.log_message.str.extract(r'Memory usage.*Current: (\d+(?:\.\d+)?)%')[0])
    frame['percent_out_of_0_100'] = frame.logged_cpu_pct.gt(100) | frame.logged_memory_pct.gt(100)
    families = frame.groupby('normalized_message', sort=True).agg(rows=('log_message', 'size'), example=('log_message', 'first'), first_csv_row=('csv_row_number', 'first')).reset_index()
    families.to_csv(out / 'log_message_families.csv', index=False)
    flags = frame.exact_duplicate_after_first | frame.level_mismatch | frame.percent_out_of_0_100
    frame.loc[flags].to_csv(out / 'flagged_logs.csv', index=False)
    severity = pd.crosstab(frame.log_level, frame.message_level)
    severity.to_csv(out / 'log_level_crosscheck.csv')
    txn_ids = frame.log_message.str.extract(r'\b(TXN\d+)\b')[0].dropna()
    metric_index = pd.DatetimeIndex(infra.timestamp)
    minute_counts = frame.assign(payment_failure=frame.log_message.str.contains(r'\b(?:NEFT|IMPS|RTGS) transaction failed\.', regex=True))
    log_counts = minute_counts.groupby('timestamp').agg(all_logs=('log_message', 'size'), payment_failure_logs=('payment_failure', 'sum')).reindex(metric_index, fill_value=0)
    log_counts.index.name = 'timestamp'
    log_counts.to_csv(out / 'log_counts_per_minute.csv')
    matching_infra = infra.set_index('timestamp')
    per_channel = []
    for channel in CHANNELS:
        counts = frame.loc[frame.log_message.str.contains(f'{channel.upper()} transaction failed.', regex=False)].groupby('timestamp').size().reindex(metric_index, fill_value=0)
        per_channel.append({'channel': channel.upper(), 'logged_failure_rows': int(counts.sum()),
                            'failure_log_count_vs_non_success_pct': pair_summary(counts, 100 - matching_infra[f'{channel}_success_rate'])})
    percentages = {}
    for column in ['logged_cpu_pct', 'logged_memory_pct']:
        s = frame[column].dropna()
        percentages[column] = {'count': len(s), 'min': float(s.min()), 'max': float(s.max()), 'above_100': int(s.gt(100).sum())}
    return {
        'family_count': len(families), 'message_families': families.to_dict('records'),
        'exact_duplicate_rows_after_first': int(frame.exact_duplicate_after_first.sum()),
        'severity_mismatches': int(frame.level_mismatch.sum()),
        'percentages': percentages, 'flagged_log_rows': int(flags.sum()),
        'logs_outside_infra_timestamp_index': int((~frame.timestamp.isin(infra.timestamp)).sum()),
        'logged_transaction_id_rows': len(txn_ids), 'distinct_logged_transaction_ids': int(txn_ids.nunique()),
        'exact_logged_transaction_id_matches_to_train': int(txn_ids.isin(txn.transaction_id).sum()),
        'cpu_vs_payment_failure_log_count': pair_summary(matching_infra.cpu_utilization, log_counts.payment_failure_logs),
        'cpu_vs_payment_failure_log_count_after_exact_deduplication': pair_summary(matching_infra.cpu_utilization, minute_counts.loc[~frame.exact_duplicate_after_first].groupby('timestamp').payment_failure.sum().reindex(metric_index, fill_value=0)),
        'per_channel_log_vs_metric': per_channel,
        'limitations': 'No host identifier in infrastructure rows and no exact transaction-ID joins. Log counts are not payment failure rates. Zero-filled bins mean no recorded log, not proof of no failure.',
        'normalization': 'Regex family counting only. These are NOT trained clusters; semantic cluster selection is deferred.',
    }


def audit_supporting(data_dir: Path, infra: pd.DataFrame, out: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    incident_path = data_dir / 'incident_reports.csv'
    if incident_path.exists():
        incidents = load(incident_path)
        shared = [c for c in incidents.select_dtypes('number').columns if c in infra.columns]
        joined = incidents.merge(infra, on='timestamp', how='inner', suffixes=('_incident', '_infra'), validate='one_to_one')
        differences = []
        mismatch_flags = pd.Series(False, index=joined.index)
        per_column = {}
        for column in shared:
            unequal = ~np.isclose(joined[f'{column}_incident'], joined[f'{column}_infra'], rtol=0, atol=1e-9)
            per_column[column] = int(unequal.sum())
            mismatch_flags |= unequal
            for idx in joined.index[unequal]:
                row = joined.loc[idx]
                differences.append({'incident_id': row.incident_id, 'timestamp': str(row.timestamp),
                                    'is_critical': int(row.is_critical), 'column': column,
                                    'incident_value': row[f'{column}_incident'], 'monitoring_value': row[f'{column}_infra']})
        pd.DataFrame(differences).to_csv(out / 'incident_metric_disagreements.csv', index=False)
        class_ranges = incidents.groupby('is_critical')[shared].agg(['min', 'max'])
        class_ranges.to_csv(out / 'incident_feature_ranges.csv')
        result['incidents'] = {
            'file_sha256': hashlib.sha256(incident_path.read_bytes()).hexdigest(),
            'rows': len(incidents), 'matching_timestamps': len(joined),
            'critical_counts': {str(k): int(v) for k, v in incidents.is_critical.value_counts().items()},
            'severity_counts': incidents.severity.value_counts().to_dict(),
            'incident_type_counts': incidents.incident_type.value_counts().to_dict(),
            'mismatched_rows': int(mismatch_flags.sum()), 'mismatched_cells': len(differences),
            'mismatches_by_column': per_column,
            'critical_mismatched_rows': int((mismatch_flags & joined.is_critical.eq(1)).sum()),
            'noncritical_mismatched_rows': int((mismatch_flags & joined.is_critical.eq(0)).sum()),
            'cpu_gt_85_rule_matches': int(incidents.cpu_utilization.gt(85).astype(int).eq(incidents.is_critical).sum()),
            'rule_is_model_evaluation': False,
            'interpretation': 'Observed separation and cross-file disagreement; generation provenance is unavailable. Do not treat deterministic rule reproduction as independently validated predictive performance.',
        }
    else:
        result['incidents'] = {'status': 'not_supplied'}
    paths = [data_dir / f'infrastructure_metrics_{part}.csv' for part in ['train', 'test']]
    if all(path.exists() for path in paths):
        train, test = [load(path) for path in paths]
        result['infrastructure_split'] = {
            'train_rows': len(train), 'test_rows': len(test),
            'train_start': str(train.timestamp.min()), 'train_end': str(train.timestamp.max()),
            'test_start': str(test.timestamp.min()), 'test_end': str(test.timestamp.max()),
            'timestamp_overlap': int(train.timestamp.isin(test.timestamp).sum()),
            'train_strictly_before_test': bool(train.timestamp.max() < test.timestamp.min()),
            'concatenation_exactly_equals_full': bool(pd.concat([train, test], ignore_index=True).equals(infra)),
            'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        }
    else:
        result['infrastructure_split'] = {'status': 'not_supplied'}
    return result


def write_report(profiles: list[dict], audit: dict, out: Path) -> None:
    monitoring_profile = next(p for p in profiles if p['file'] == 'infrastructure_metrics.csv')
    monitoring_rows = monitoring_profile['rows']
    total_missing = sum(c['missing'] for p in profiles for c in p['schema'])
    error_summary = next(c for c in monitoring_profile['schema'] if c['column'] == 'application_error_rate')['numeric_summary']
    lines = ['# Dataset discovery', '',
             'Generated by `scripts/analyze_discovery.py`. Values are computed from the supplied files. '
             'No source rows were changed, no imputation was performed, and no models were trained.', '',
             '## Source schema', '']
    for profile in profiles:
        lines += [f"### {profile['file']}", '', f"Rows: {profile['rows']:,}; columns: {profile['columns']}; exact duplicate rows after first: {profile['duplicate_rows']}.", '',
                  '| Column | CSV-inferred dtype | Missing | Unique non-null |', '|---|---|---:|---:|']
        for c in profile['schema']:
            lines.append(f"| `{c['column']}` | `{c['csv_inferred_dtype']}` | {c['missing']} | {c['unique_non_null']} |")
        for c in profile['schema']:
            if 'parsed_dtype' in c:
                lines += ['', f"Timestamp: {c['min']} through {c['max']}; parsed dtype `{c['parsed_dtype']}`; timezone `{c['timezone']}`; invalid {c['invalid_timestamps']}; sorted {c['monotonic_increasing']}."]
            if 'value_counts' in c:
                counts = '; '.join(f'{key}: {value}' for key, value in c['value_counts'].items())
                lines += ['', f"**{c['column']}**: {counts}."]
        lines += ['']
    lines += ['## Operational failure is not fraud', '',
              'The transaction CSV has `is_fraud` and `processing_time_ms`, but no operational outcome. '
              'The infrastructure CSV has NEFT/IMPS/RTGS success percentages. We derive the complement '
              '`100 - success_rate` separately for each channel. This is a non-success percentage, '
              'not a count of failed payments. Pending/timeout treatment and channel denominators are not supplied. '
              'No global volume-weighted success rate is identifiable from these files.', '',
              'There is no `api_latency_ms`, endpoint-level latency series, host key, or channel-level latency '
              'in the infrastructure CSV. Aggregate processing time must not be relabeled as API or channel latency.', '',
              '## Same-minute relationships', '',
              f'Pearson and rank-based Spearman coefficients below use all {monitoring_rows:,} minute observations. '
              'These are descriptive, not causal, and no independence-based p-values are reported. '
              '15-minute/hourly aggregation and first-difference/hour-of-day sensitivities are in the companion CSVs.', '',
              '| Load feature | NEFT non-success r | IMPS non-success r | RTGS non-success r | Aggregate processing time r |',
              '|---|---:|---:|---:|---:|']
    corrs = audit['infrastructure']['minute_correlations']
    for feature in LOAD_COLUMNS:
        lookup = {c['target']: c['pearson_r'] for c in corrs if c['feature'] == feature}
        vals = [lookup[f'{ch}_non_success_pct'] for ch in CHANNELS] + [lookup[LATENCY]]
        lines.append('| `' + feature + '` | ' + ' | '.join(f'{v:.6f}' for v in vals) + ' |')
    lines += ['', '## High-load comparison', '',
              'High load here means CPU strictly above the full-sample 95th percentile. '
              'This percentile is an exploratory contrast, not a production alert rule or an SLO. '
              'Rate differences are percentage points; processing differences are milliseconds.', '',
              '| Metric | High CPU mean | Other minutes mean | Difference |', '|---|---:|---:|---:|']
    for row in audit['infrastructure']['high_cpu_comparison']:
        lines.append(f"| `{row['target']}` | {row['high_cpu_mean']:.6f} | {row['remaining_mean']:.6f} | {row['difference']:.6f} |")
    example = audit['infrastructure']['high_cpu_comparison'][0]
    lines += ['', f"CPU threshold: {example['cpu_p95_threshold']:.6f}; high-load minutes: {example['high_cpu_minutes']}; remaining: {example['remaining_minutes']}.", '',
              '## Timestamp alignment', '', '```json', json.dumps(audit['alignment'], indent=2), '```', '',
              '## Actual anomalies and constraints', '']
    infra = audit['infrastructure']; logs = audit['logs']
    lines += [f"- Parsed missing values across the {len(profiles)} primary files: {total_missing}. `latency == 100ms` occurs in {infra['latency_exactly_100ms_minutes']:,} monitoring rows; this is an observed pile-up, not automatically an error.",
              f"- CPU exceeds 85 in {infra['cpu_above_85_minutes']} monitoring rows. A CPU-85 breach must not be invented in this history.",
              f"- Logs contain {logs['exact_duplicate_rows_after_first']} exact duplicate rows after the first, and {logs['severity_mismatches']} structured/message severity disagreements. Preserve raw records; flag rather than silently overwrite or remove.",
              '- Logged resource percentages: `' + json.dumps(logs['percentages']) + '`. These are outside a conventional 0–100 metric scale. CPU scope may differ (per-process/per-core versus host); neither host-level mapping nor a measurement definition is provided. Memory percentages also require validation, not clipping.',
              f"- Regex masking produces {logs['family_count']} distinct message families. These are not learned clusters or evidence for choosing the future cluster count.",
              f"- Exact log transaction-ID matches to the transaction training file: {logs['exact_logged_transaction_id_matches_to_train']}. No trace linkage is demonstrated.",
              f"- `application_error_rate` contains integers {error_summary['min']:g}–{error_summary['max']:g}, but its denominator and measurement window are undocumented. Do not display it as a percentage.",
              '- `transaction_volume_per_hour` appears on each minute row. Do not sum these values into total payments without the aggregation definition.',
              '- Amount currency, timestamp timezone, source provenance, and permitted redistribution are not established by these files.', '',
              '**Non-100 service-uptime rows:**', '', '```json', json.dumps(infra['non_100_uptime_records'], indent=2), '```', '',
              '**Time-of-day regimes:**', '', '```json', json.dumps(infra['time_of_day_regimes'], indent=2), '```', '',
              '## Log message families', '', '| Normalized message | Rows | First source CSV row |', '|---|---:|---:|']
    for family in logs['message_families']:
        lines.append(f"| `{family['normalized_message']}` | {family['rows']} | {family['first_csv_row']} |")
    lines += ['', '## Log cross-checks', '', '```json', json.dumps({k:v for k,v in logs.items() if k not in ('message_families', 'percentages')}, indent=2), '```', '',
              '## Supporting-file audit (not used to rewrite monitoring history)', '',
              '```json', json.dumps(audit['supporting'], indent=2), '```', '',
              '## Boundary for the next phase', '',
              'Full-sample exploration is disclosed. Keep the supplied chronological holdout fixed; '
              'fit preprocessing, alert thresholds and hyperparameters on training folds only. '
              'A five-minute forecast target, lag/rolling features and a daily seasonal baseline are proposals, '
              'not implemented features. Any rolling feature must use only information available at the forecast origin.', '',
              'The incident audit compares labels and metric values at matching timestamps when '
              'incident records are supplied. Rule agreement is a data diagnostic, not validated '
              'model accuracy. Investigate label provenance and distinguish current-state triage '
              'from future-incident prediction.', '',
              'This report covers data discovery only; it does not evaluate models or application services.', '']
    (out / 'discovery_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    missing = [name for name in FILES if not (args.data_dir / name).is_file()]
    if missing:
        parser.error('Missing input files: ' + ', '.join(missing))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    txn, infra, logs = [load(args.data_dir / name) for name in FILES]
    if not infra.timestamp.is_unique:
        raise ValueError('Monitoring timestamps are not unique: minute-level alignment is ambiguous')
    profiles = [inspect(args.data_dir / name) for name in FILES]
    audit = {
        'environment': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__},
        'source_sha256': {p['file']: p['sha256'] for p in profiles},
        'infrastructure': audit_relationships(infra, out),
        'alignment': audit_alignment(txn, infra, out),
        'logs': audit_logs(logs, txn, infra, out),
        'supporting': audit_supporting(args.data_dir, infra, out),
    }
    (out / 'schema.json').write_text(json.dumps(profiles, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (out / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_report(profiles, audit, out)
    print('Data discovery complete. Source CSVs unchanged; no models trained.')
    print('\nSAME-MINUTE CORRELATIONS (all source monitoring rows)')
    print(pd.DataFrame(audit['infrastructure']['minute_correlations']).to_string(index=False))
    print('\nHIGH-CPU CONTRAST (exploratory full-sample P95, not an alert threshold)')
    print(pd.DataFrame(audit['infrastructure']['high_cpu_comparison']).to_string(index=False))
    print('\nALIGNMENT\n' + json.dumps(audit['alignment'], indent=2))
    print('\nSUPPORTING-FILE CHECKS\n' + json.dumps(audit['supporting'], indent=2))
    print('\nDetailed report: ' + str(out / 'discovery_report.md'))


if __name__ == '__main__':
    main()
