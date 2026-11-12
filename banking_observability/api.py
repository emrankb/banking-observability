"""Local API for recorded banking metrics and the project's saved models."""
from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from .detection_common import MONITORING_METRICS
from .incident_evidence import read_evidence
from .kpi_forecasting import TARGETS

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CLOCK = 'timezone-naive source timestamps; timezone unspecified'
TargetName = Literal[
    'cpu_utilization', 'memory_usage', 'db_connection_pool_usage',
    'neft_success_rate', 'imps_success_rate', 'rtgs_success_rate',
]
SourceKind = Literal['runbooks', 'supplied_runbook', 'project_runbook', 'incident_snapshot', 'all']


class FixedInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class Origin(FixedInput):
    as_of: str = Field(min_length=19, max_length=26)


class TargetOrigin(Origin):
    target: TargetName


class LogMessage(FixedInput):
    message: str = Field(min_length=1, max_length=4096)
    log_level: Literal['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG']


class KnowledgeQuery(FixedInput):
    query: str = Field(min_length=1, max_length=1000)
    source_kind: SourceKind = 'runbooks'
    top_k: int = Field(default=3, ge=1, le=10)
    min_similarity: float = Field(default=0.15, ge=0, le=1)
    as_of: str | None = Field(default=None, max_length=26)


class Investigation(Origin):
    question: str = Field(min_length=1, max_length=1000)
    include_historical_incidents: bool = False


def source_minute(value: str) -> pd.Timestamp:
    """Accept only whole minutes on the timestamps' undeclared source clock."""
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', value):
        raise ValueError('Expected an ISO-8601 minute with seconds, without a timezone.')
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Expected a recorded ISO-8601 minute, without a timezone.') from exc
    if parsed.tzinfo is not None or parsed.second or parsed.microsecond:
        raise ValueError('The source clock uses timezone-naive whole-minute timestamps.')
    return pd.Timestamp(parsed)


def _json_number(value: object) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError('Non-finite source metric.')
    return number


def read_metric_range(path: Path, *, start: str | None, end: str | None, limit: int) -> dict:
    """Return a bounded slice of the source CSV without interpolating missing minutes."""
    start_time = source_minute(start) if start is not None else None
    end_time = source_minute(end) if end is not None else None
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError('start must be at or before end.')
    frame = pd.read_csv(path)
    if not {'timestamp', *MONITORING_METRICS}.issubset(frame.columns):
        raise ValueError('The monitoring CSV has missing metric columns.')
    times = pd.to_datetime(frame['timestamp'], format='ISO8601', errors='raise')
    if times.isna().any() or times.dt.tz is not None or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError('The source has invalid or unordered timestamps.')
    if not times.eq(times.dt.floor('min')).all():
        raise ValueError('Source measurements must be at whole-minute timestamps.')
    mask = pd.Series(True, index=frame.index)
    if start_time is not None:
        mask &= times.ge(start_time)
    if end_time is not None:
        mask &= times.le(end_time)
    subset = frame.loc[mask]
    if subset.empty:
        raise ValueError('No monitoring records match the requested interval.')
    omitted = max(0, len(subset) - limit)
    subset = subset.tail(limit)
    selected = subset.loc[:, list(MONITORING_METRICS)].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError('Source metrics contain non-finite values.')
    rows = [
        {'timestamp': times.loc[idx].isoformat(),
         'metrics': {name: _json_number(value) for name, value in record.items()}}
        for idx, record in selected.iterrows()
    ]
    return {
        'mode': 'historical_replay', 'source_timezone': SOURCE_CLOCK,
        'metric_names': list(MONITORING_METRICS), 'count': len(rows),
        'older_matching_rows_omitted': omitted, 'rows': rows,
    }


def create_app(root: Path | None = None) -> FastAPI:
    """Bind data and artifact locations at startup; clients cannot choose paths."""
    project = (Path(root) if root is not None else ROOT).resolve()
    data = project / 'data' / 'raw'
    models = project / 'models'
    api = FastAPI(title='Banking Observability API', version='0.1.0',
                  description='Local historical replay of stored banking operations data and fitted models.')

    def call(operation):
        try:
            return operation()
        except (FileNotFoundError, ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail='Local data, dependency, or model artifact is unavailable.') from exc
        except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
            raise HTTPException(status_code=422, detail='Invalid input, source data, or incompatible model artifact.') from exc

    def source_frame() -> pd.DataFrame:
        return pd.read_csv(data / 'infrastructure_metrics.csv')

    @api.get('/api/v1/health')
    def health():
        artifacts = {
            'latency_forecast': (models / 'latency_5min' / 'metadata.json').is_file(),
            'kpi_forecasts': {name: (models / 'kpi_5min' / name / 'metadata.json').is_file() for name in TARGETS},
            'monitoring_detector': (models / 'detection' / 'monitoring_anomaly' / 'metadata.json').is_file(),
            'incident_triage': (models / 'detection' / 'incident_triage' / 'metadata.json').is_file(),
            'log_classifier': (models / 'log_clustering' / 'metadata.json').is_file(),
            'knowledge_index': (models / 'knowledge' / 'manifest.json').is_file(),
        }
        return {'status': 'ok', 'mode': 'historical_replay', 'external_generation': False,
                'source_timezone': SOURCE_CLOCK, 'artifacts_present': artifacts}

    @api.get('/api/v1/metrics')
    def metrics(start: str | None = None, end: str | None = None,
                limit: int = Query(default=120, ge=1, le=720)):
        return call(lambda: read_metric_range(data / 'infrastructure_metrics.csv',
                                              start=start, end=end, limit=limit))

    @api.get('/api/v1/metrics/summary')
    def metric_summary(as_of: str):
        def get():
            origin = source_minute(as_of)
            packet = read_evidence(data, as_of=origin.isoformat(), lookback_minutes=1)
            return {'as_of': packet['as_of'], 'metrics': packet['monitoring'],
                    'baseline_minutes': 60, 'mode': 'historical_replay'}
        return call(get)

    @api.post('/api/v1/forecasts/latency')
    def latency(request: Origin):
        def predict():
            from .forecasting import LatencyForecaster
            return LatencyForecaster(models / 'latency_5min').predict(
                source_frame(), as_of=source_minute(request.as_of))
        return call(predict)

    @api.post('/api/v1/forecasts/kpi')
    def forecast_kpi(request: TargetOrigin):
        def predict():
            from .kpi_forecasting import KPIForecaster
            return KPIForecaster(models / 'kpi_5min' / request.target).predict(
                source_frame(), as_of=source_minute(request.as_of))
        return call(predict)

    @api.post('/api/v1/monitoring/anomaly')
    def anomaly(request: Origin):
        def score():
            from .anomaly_detection import MonitoringDetector
            return MonitoringDetector(models / 'detection' / 'monitoring_anomaly').predict(
                source_frame(), as_of=source_minute(request.as_of))
        return call(score)

    @api.post('/api/v1/logs/classify')
    def classify(request: LogMessage):
        def label():
            from .log_clustering import LogClassifier
            result = LogClassifier(models / 'log_clustering').predict(request.message, request.log_level)
            return result  # The response contains normalized wording, not the original message.
        return call(label)

    @api.get('/api/v1/logs/recent')
    def recent_logs(as_of: str, lookback_minutes: int = Query(default=30, ge=1, le=120),
                    limit: int = Query(default=20, ge=1, le=20)):
        def read():
            from .log_clustering import LogClassifier
            origin = source_minute(as_of)
            packet = read_evidence(data, as_of=origin.isoformat(), lookback_minutes=lookback_minutes)
            classifier = LogClassifier(models / 'log_clustering')
            rows = []
            for log in packet['logs_most_recent_first'][:limit]:
                result = classifier.predict(log['log_message'], log['log_level'])
                rows.append({
                    'timestamp': log['timestamp'], 'csv_row': log['csv_row'],
                    'normalized_message': log['normalized_message'], 'log_level': log['log_level'],
                    'event_type': result['event_type'], 'category': result['category'],
                    'label_review_required': result['label_review_required'],
                    'quality_flags': result['quality_flags'], 'reference': log['reference'],
                })
            return {'as_of': packet['as_of'], 'lookback_minutes': lookback_minutes,
                    'matching_source_rows': packet['recent_log_count'], 'rows': rows}
        return call(read)

    @api.post('/api/v1/knowledge/search')
    def knowledge(request: KnowledgeQuery):
        def search():
            from .knowledge_retrieval import KnowledgeStore
            store = KnowledgeStore(models / 'knowledge')
            if store.source_changes(data):
                raise ValueError('Indexed source files changed.')
            return store.search(request.query, top_k=request.top_k,
                                source_kind=request.source_kind, min_similarity=request.min_similarity,
                                as_of=source_minute(request.as_of).isoformat() if request.as_of else None)
        return call(search)

    @api.get('/api/v1/incidents/{incident_id}/triage')
    def incident_triage(incident_id: str):
        def score():
            from .incident_triage import IncidentTriage
            if len(incident_id) > 64 or not incident_id.startswith('INC') or not incident_id[3:].isdigit():
                raise ValueError('Expected an incident ID from the supplied incident file.')
            records = pd.read_csv(data / 'incident_reports.csv')
            matched = records.loc[records['incident_id'].eq(incident_id)]
            if len(matched) != 1:
                raise ValueError('Incident record not found.')
            return {'scope': 'supplied_incident_snapshot_only',
                    'result': IncidentTriage(models / 'detection' / 'incident_triage').predict(matched.iloc[0].to_dict())}
        return call(score)

    @api.post('/api/v1/investigations')
    def investigate(request: Investigation):
        def build():
            from .incident_assistant import run_assistant
            return run_assistant(
                question=request.question, as_of=source_minute(request.as_of).isoformat(),
                data_dir=data, knowledge_dir=models / 'knowledge',
                log_model_dir=models / 'log_clustering',
                anomaly_model_dir=models / 'detection' / 'monitoring_anomaly',
                include_historical=request.include_historical_incidents,
                model=None, allow_external_data=False,
            )
        return call(build)

    return api


app = create_app()
