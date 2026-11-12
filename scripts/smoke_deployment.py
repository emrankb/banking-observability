#!/usr/bin/env python3
"""Exercise the running local API and dashboard against original source records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = '2025-10-25T18:01:00'
FORECAST_ORIGIN = '2025-10-25T14:24:00'
TARGET_TIME = '2025-10-25T14:29:00'
MODEL_TARGETS = (
    'cpu_utilization', 'memory_usage', 'db_connection_pool_usage',
    'neft_success_rate', 'imps_success_rate', 'rtgs_success_rate',
)


class SmokeFailure(RuntimeError):
    """An observed response did not match the source or endpoint contract."""


def local_origin(value: str) -> str:
    """Allow only explicit local HTTP origins; no remote API addresses."""
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1'}
            or parsed.username or parsed.password or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment or parsed.port is None
            or not 1 <= parsed.port <= 65535):
        raise ValueError('Use a local HTTP origin such as http://127.0.0.1:8000.')
    return value.rstrip('/')


def get_json(url: str, *, body: dict | None = None) -> dict:
    request = Request(url, method='POST' if body is not None else 'GET',
                      data=json.dumps(body, allow_nan=False).encode('utf-8') if body is not None else None,
                      headers={'Content-Type': 'application/json'} if body is not None else {})
    with urlopen(request, timeout=25) as response:
        if response.status != 200:
            raise SmokeFailure(f'Unexpected HTTP status: {response.status}')
        result = json.load(response)
    if not isinstance(result, dict):
        raise SmokeFailure('Expected a JSON object.')
    return result


def source_minute(data_dir: Path, minute: str = ORIGIN) -> tuple[int, dict[str, str]]:
    """Return the actual source CSV row and physical row number (header is row 1)."""
    with (data_dir / 'infrastructure_metrics.csv').open(newline='', encoding='utf-8') as f:
        rows = csv.DictReader(f)
        found = [(n, r) for n, r in enumerate(rows, 2) if r['timestamp'].replace(' ', 'T') == minute]
    if len(found) != 1:
        raise SmokeFailure('Expected exactly one source monitoring row for the replay minute.')
    return found[0]


def require(condition: object, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def verify_metrics(body: dict, recorded: dict[str, str]) -> None:
    require(body.get('mode') == 'historical_replay', 'Metrics endpoint is not historical replay.')
    require(body.get('count') == 1 and len(body.get('rows', [])) == 1,
            'Expected exactly one source monitoring observation.')
    row = body['rows'][0]
    require(row['timestamp'] == ORIGIN, 'Monitoring timestamp differs from the requested minute.')
    for name, value in row['metrics'].items():
        require(name in recorded, f'Monitoring metric absent from source: {name}')
        require(math.isclose(float(value), float(recorded[name]), rel_tol=0, abs_tol=1e-9),
                f'Monitoring metric differs from source: {name}')


def verify_reference(root: Path, reference: dict) -> None:
    path = reference.get('path')
    require(isinstance(path, str) and bool(path), 'Missing source reference path.')
    relative = Path(path)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Source path is not repository-relative.')
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(),
            f'Referenced source is not present: {path}')
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    require(digest == reference.get('sha256'), f'Source checksum mismatch: {path}')
    if 'csv_row' in reference:
        require(isinstance(reference['csv_row'], int) and reference['csv_row'] >= 2,
                'Invalid CSV row reference.')
    else:
        require(isinstance(reference.get('line_start'), int) and
                isinstance(reference.get('line_end'), int) and
                1 <= reference['line_start'] <= reference['line_end'],
                'Missing runbook line range.')


def smoke(*, root: Path, api_url: str, dashboard_url: str) -> list[str]:
    api_url = local_origin(api_url)
    dashboard_url = local_origin(dashboard_url)
    data = root / 'data/raw'
    row_number, actual = source_minute(data)
    results = []

    health = get_json(api_url + '/api/v1/health')
    status = health.get('artifacts_present', {})
    present = [item for value in status.values() for item in
               (value.values() if isinstance(value, dict) else (value,))]
    require(health.get('status') == 'ok' and health.get('mode') == 'historical_replay'
            and health.get('external_generation') is False and bool(present)
            and all(x is True for x in present), 'The API does not report all local artifacts.')
    results.append('API health and saved artifacts')

    metric = get_json(api_url + '/api/v1/metrics?start=' + ORIGIN + '&end=' + ORIGIN + '&limit=1')
    verify_metrics(metric, actual)
    results.append('Recorded monitoring values equal original CSV')

    summary = get_json(api_url + '/api/v1/metrics/summary?as_of=' + ORIGIN)
    require(summary.get('as_of') == ORIGIN and summary.get('baseline_minutes') == 60,
            'Monitoring summary has the wrong origin or reference window.')
    require(any(row.get('metric') == 'application_error_rate'
                and math.isclose(float(row['observed']), float(actual['application_error_rate']))
                for row in summary.get('metrics', [])), 'Monitoring summary does not match the source.')
    results.append('Past-only monitoring summary')

    latency = get_json(api_url + '/api/v1/forecasts/latency', body={'as_of': FORECAST_ORIGIN})
    require(latency.get('forecast_origin') == FORECAST_ORIGIN
            and latency.get('target_timestamp') == TARGET_TIME
            and latency.get('horizon_minutes') == 5
            and 'actual' not in latency, 'Latency forecast does not respect its t+5 contract.')
    results.append('Saved processing-time forecast')

    for target in MODEL_TARGETS:
        reply = get_json(api_url + '/api/v1/forecasts/kpi',
                         body={'as_of': FORECAST_ORIGIN, 'target': target})
        require(reply.get('target_column') == target
                and reply.get('target_timestamp') == TARGET_TIME
                and reply.get('horizon_minutes') == 5 and 'actual' not in reply
                and math.isfinite(float(reply['predicted_value'])),
                f'Invalid saved forecast for {target}.')
    results.append('Six saved resource and payment-channel forecasts')

    anomaly = get_json(api_url + '/api/v1/monitoring/anomaly', body={'as_of': ORIGIN})
    require(anomaly.get('timestamp') == ORIGIN and isinstance(anomaly.get('review_required'), bool)
            and math.isfinite(float(anomaly['anomaly_score'])),
            'Anomaly result does not correspond to the selected minute.')
    results.append('Saved anomaly detector')

    logs = get_json(api_url + '/api/v1/logs/recent?as_of=' + ORIGIN + '&lookback_minutes=30')
    require(logs.get('as_of') == ORIGIN and bool(logs.get('rows')),
            'Recent log query returned no source logs.')
    for log in logs['rows']:
        require(log['timestamp'] <= ORIGIN and 'log_message' not in log,
                'Log list leaked future data or original unmasked wording.')
        verify_reference(root, log['reference'])
    results.append('Recent logs and original source coordinates')

    with (data / 'application_logs.csv').open(newline='', encoding='utf-8') as f:
        first = next(csv.DictReader(f))
    classified = get_json(api_url + '/api/v1/logs/classify',
                          body={'message': first['log_message'], 'log_level': first['log_level']})
    require(classified.get('template_seen_during_fit') is True and
            'log_message' not in classified, 'Known log classification is missing or unmasked.')
    results.append('Saved log classifier')

    found = get_json(api_url + '/api/v1/knowledge/search',
                     body={'query': 'network timeout latency', 'as_of': ORIGIN})
    require(found.get('status') == 'matches' and bool(found.get('results')),
            'Runbook retrieval returned no matching documents.')
    require(found['results'][0].get('document_id') == 'runbook:network_timeout',
            'Network timeout did not retrieve its expected runbook.')
    for passage in found['results']:
        for reference in passage.get('references', []):
            verify_reference(root, reference)
    results.append('FAISS search and checked runbook citations')

    incident = get_json(api_url + '/api/v1/incidents/INC000136/triage')
    require(incident.get('scope') == 'supplied_incident_snapshot_only'
            and bool(incident.get('result')), 'Historical incident record triage failed.')
    results.append('Historical incident triage')

    packet = get_json(api_url + '/api/v1/investigations',
                      body={'as_of': ORIGIN, 'question': 'network timeout latency'})
    require(packet.get('status') == 'evidence_only' and packet.get('as_of') == ORIGIN
            and packet.get('root_cause') is None and packet.get('resolution') is None
            and packet.get('model_explanation') is None,
            'Investigation response is not the local evidence-only contract.')
    require(packet.get('monitoring_metrics') == 10 and packet.get('runbook_passages', 0) >= 1,
            'Investigation packet omitted monitoring or runbook evidence.')
    for item in packet.get('runbook_evidence', []):
        verify_reference(root, item['source'])
    require(any(item.get('source', {}).get('csv_row') == row_number
                for item in packet.get('monitoring_evidence', [])),
            'Investigation metric references do not match the source minute.')
    results.append('LangGraph evidence-only investigation')

    with urlopen(dashboard_url + '/_stcore/health', timeout=10) as response:
        require(response.status == 200 and response.read(64).strip() == b'ok',
                'Streamlit health endpoint did not respond with ok.')
    results.append('Streamlit HTTP health')
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-url', default='http://127.0.0.1:18000')
    parser.add_argument('--dashboard-url', default='http://127.0.0.1:18501')
    parser.add_argument('--project-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        checks = smoke(root=args.project_dir.resolve(), api_url=args.api_url,
                       dashboard_url=args.dashboard_url)
    except (ValueError, KeyError, TypeError, OSError, HTTPError, URLError, SmokeFailure) as exc:
        print(f'Deployment smoke check failed: {exc}', file=sys.stderr)
        return 1
    for check in checks:
        print(f'OK: {check}')
    print(f'Passed {len(checks)} deployment checks using original records.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
