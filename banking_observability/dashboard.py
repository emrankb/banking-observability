"""HTTP access and display transforms for the historical monitoring dashboard."""
from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Any
from urllib.parse import urlsplit

import httpx
import pandas as pd


DEFAULT_API_URL = "http://127.0.0.1:8000"
SOURCE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
KPI_GROUPS = {
    "Processing time · ms": ("avg_transaction_processing_time_ms",),
    "Infrastructure and errors · %": (
        "cpu_utilization", "memory_usage", "db_connection_pool_usage",
        "application_error_rate", "service_uptime",
    ),
    "Payment and authentication · %": (
        "neft_success_rate", "imps_success_rate", "rtgs_success_rate", "auth_success_rate",
    ),
    "Network throughput · Mbps": ("network_throughput_mbps",),
    "Transaction volume · per hour": ("transaction_volume_per_hour",),
}
FORECAST_TARGETS = {
    "Processing time (ms)": "avg_transaction_processing_time_ms",
    "CPU utilization (%)": "cpu_utilization",
    "Memory usage (%)": "memory_usage",
    "DB pool utilization (%)": "db_connection_pool_usage",
    "NEFT success rate (%)": "neft_success_rate",
    "IMPS success rate (%)": "imps_success_rate",
    "RTGS success rate (%)": "rtgs_success_rate",
}


class DashboardError(Exception):
    """A local API request failed; the response is not suitable for display."""


def recorded_minute(value: str) -> str:
    """Use the source's exact timezone-naive minute representation."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00", value):
        raise ValueError("Select a recorded whole minute without a timezone.")
    try:
        dt = datetime.strptime(value, SOURCE_TIME_FORMAT)
    except ValueError as exc:
        raise ValueError("Invalid date or time.") from exc
    if dt.strftime(SOURCE_TIME_FORMAT) != value:
        raise ValueError("Invalid source timestamp.")
    return value


def source_date_time(day, minute) -> str:
    return recorded_minute(datetime.combine(day, minute.replace(second=0, microsecond=0)).strftime(SOURCE_TIME_FORMAT))


def validate_api_url(value: str) -> str:
    """API location is configured by the operator, never by a dashboard visitor."""
    parts = urlsplit(value)
    if (parts.scheme != "http" or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or parts.path not in ("", "/")):
        raise ValueError("BANKING_API_URL must be a plain http://host:port origin.")
    return value.rstrip("/")


class DashboardAPI:
    def __init__(self, url: str | None = None, *, transport: httpx.BaseTransport | None = None):
        configured = url if url is not None else os.environ.get("BANKING_API_URL", DEFAULT_API_URL)
        self.url = validate_api_url(configured)
        self.http = httpx.Client(base_url=self.url, timeout=20.0, trust_env=False, transport=transport)

    def close(self) -> None:
        self.http.close()

    def _request(self, method: str, path: str, *, params: dict | None = None,
                 payload: dict | None = None) -> dict:
        try:
            response = self.http.request(method, path, params=params, json=payload)
        except httpx.RequestError as exc:
            raise DashboardError("Cannot reach the local FastAPI service. Start scripts/serve_api.py first.") from exc
        if response.status_code >= 400:
            if response.status_code == 422:
                raise DashboardError("The API rejected this time or input. Choose a supported recorded minute.")
            if response.status_code == 503:
                raise DashboardError("The API is missing local source files or saved model artifacts.")
            raise DashboardError(f"The local API returned HTTP {response.status_code}.")
        try:
            body = response.json()
        except ValueError as exc:
            raise DashboardError("The local API returned non-JSON data.") from exc
        if not isinstance(body, dict):
            raise DashboardError("Unexpected API response shape.")
        return body

    def health(self) -> dict:
        return self._request("GET", "/api/v1/health")

    def metrics(self, *, end: str, limit: int = 240) -> dict:
        if isinstance(limit, bool) or not 1 <= limit <= 720:
            raise ValueError("History limit must be between 1 and 720 minutes.")
        return self._request("GET", "/api/v1/metrics", params={"end": recorded_minute(end), "limit": limit})

    def summary(self, *, as_of: str) -> dict:
        return self._request("GET", "/api/v1/metrics/summary", params={"as_of": recorded_minute(as_of)})

    def forecast(self, *, as_of: str, target: str) -> dict:
        if target not in FORECAST_TARGETS.values():
            raise ValueError("Unknown forecast target.")
        payload: dict[str, str] = {"as_of": recorded_minute(as_of)}
        if target == "avg_transaction_processing_time_ms":
            return self._request("POST", "/api/v1/forecasts/latency", payload=payload)
        payload["target"] = target
        return self._request("POST", "/api/v1/forecasts/kpi", payload=payload)

    def anomaly(self, *, as_of: str) -> dict:
        return self._request("POST", "/api/v1/monitoring/anomaly", payload={"as_of": recorded_minute(as_of)})

    def logs(self, *, as_of: str, lookback_minutes: int = 30) -> dict:
        if not 1 <= lookback_minutes <= 120:
            raise ValueError("Log lookback must be between 1 and 120 minutes.")
        return self._request("GET", "/api/v1/logs/recent", params={
            "as_of": recorded_minute(as_of), "lookback_minutes": lookback_minutes, "limit": 20})

    def classify(self, *, message: str, level: str) -> dict:
        if not message.strip() or len(message) > 4096 or level not in {"INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"}:
            raise ValueError("Provide a log message and a valid severity level.")
        return self._request("POST", "/api/v1/logs/classify", payload={"message": message, "log_level": level})

    def knowledge(self, *, query: str, as_of: str) -> dict:
        if not query.strip() or len(query) > 1000:
            raise ValueError("Knowledge query must contain 1–1000 characters.")
        return self._request("POST", "/api/v1/knowledge/search", payload={
            "query": query.strip(), "as_of": recorded_minute(as_of), "source_kind": "runbooks"})

    def investigate(self, *, question: str, as_of: str, include_historical: bool = False) -> dict:
        if not question.strip() or len(question) > 1000:
            raise ValueError("Investigation question must contain 1–1000 characters.")
        return self._request("POST", "/api/v1/investigations", payload={
            "question": question.strip(), "as_of": recorded_minute(as_of),
            "include_historical_incidents": bool(include_historical)})


    def check_draft(self, *, question: str, as_of: str, draft: dict) -> dict:
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("An investigation question is required.")
        if not isinstance(draft, dict):
            raise ValueError("Select a JSON incident draft.")
        return self._request("POST", "/api/v1/investigations/check-draft", payload={
            "question": question.strip(), "as_of": recorded_minute(as_of), "draft": draft})


def monitoring_frame(response: dict) -> pd.DataFrame:
    """Create a time-indexed frame from original API values, without interpolation."""
    rows = response.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No recorded monitoring rows in the response.")
    if any(not isinstance(row, dict) or not isinstance(row.get("metrics"), dict) for row in rows):
        raise ValueError("Malformed monitoring records.")
    timestamps = [recorded_minute(row["timestamp"]) for row in rows]
    if timestamps != sorted(set(timestamps)):
        raise ValueError("Monitoring times must be ordered and unique.")
    values = pd.DataFrame([r["metrics"] for r in rows], index=pd.DatetimeIndex(timestamps, name="timestamp"))
    if values.empty or not all(pd.api.types.is_numeric_dtype(values[c]) for c in values.columns):
        raise ValueError("Monitoring records must contain numeric metrics.")
    return values


def recorded_minutes(response: dict) -> list[str]:
    return [recorded_minute(row["timestamp"]) for row in response.get("rows", [])]


def summary_by_metric(response: dict) -> dict[str, dict]:
    entries = response.get("metrics")
    if not isinstance(entries, list):
        raise ValueError("Missing metric summary.")
    result = {item["metric"]: item for item in entries}
    if len(result) != len(entries):
        raise ValueError("Duplicate summary metric.")
    return result


def forecast_fields(reply: dict) -> dict:
    target = reply.get("target_column")
    if target not in FORECAST_TARGETS.values():
        raise ValueError("Unexpected forecast target.")
    number_key = "prediction_ms" if target == "avg_transaction_processing_time_ms" else "predicted_value"
    if number_key not in reply:
        raise ValueError("Missing forecast value.")
    return {"target": target, "value": float(reply[number_key]), "unit": "ms" if number_key == "prediction_ms" else "%",
            "target_time": recorded_minute(reply["target_timestamp"]), "origin": recorded_minute(reply["forecast_origin"]),
            "model": reply["model_name"]}


def evidence_links(reply: dict) -> list[dict]:
    """Retain source paths and line/row locations; do not turn them into causal claims."""
    sources: list[dict] = []
    for item in reply.get("monitoring_evidence", []):
        ref = item["source"]
        sources.append({"evidence": item["id"], "file": ref["path"], "row": ref["csv_row"], "lines": None})
    for item in reply.get("recent_log_evidence", []):
        ref = item["source"]
        sources.append({"evidence": item["id"], "file": ref["path"], "row": ref["csv_row"], "lines": None})
    for item in reply.get("runbook_evidence", []):
        ref = item["source"]
        sources.append({"evidence": item["id"], "file": ref["path"], "row": None,
                        "lines": f"{ref['line_start']}–{ref['line_end']}"})
    return sources
