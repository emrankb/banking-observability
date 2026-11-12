#!/usr/bin/env python3
"""Container health probes for the two local services."""
import json
from pathlib import Path
import sys
from urllib.request import urlopen


REQUIRED_SOURCE_FILES = (
    'infrastructure_metrics.csv',
    'infrastructure_metrics_train.csv',
    'application_logs.csv',
    'incident_reports.csv',
    'incident_knowledge_base.txt',
)


def api_ready() -> bool:
    data = Path('/app/data/raw')
    if not all((data / name).is_file() for name in REQUIRED_SOURCE_FILES):
        return False
    with urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4) as response:
        if response.status != 200:
            return False
        body = json.load(response)
    artifacts = body.get('artifacts_present', {})
    available = [item for value in artifacts.values() for item in
                 (value.values() if isinstance(value, dict) else (value,))]
    return (body.get('status') == 'ok' and body.get('mode') == 'historical_replay'
            and body.get('external_generation') is False
            and bool(available) and all(item is True for item in available))


def dashboard_ready() -> bool:
    with urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4) as response:
        return response.status == 200 and response.read(64).strip() == b'ok'


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {'api', 'dashboard'}:
        print('Usage: check_container_ready.py api|dashboard', file=sys.stderr)
        return 2
    try:
        return 0 if (api_ready() if sys.argv[1] == 'api' else dashboard_ready()) else 1
    except (OSError, ValueError, KeyError, TimeoutError):
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
