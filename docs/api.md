# Historical monitoring API

The API provides read-only access to the CSV-backed monitoring history and the models already trained in this repository. It uses FastAPI with a fixed data directory (`data/raw/`) and fixed model directories (`models/`). Request bodies cannot select local paths or upload replacement model files.

This is **historical replay**, not a live banking telemetry feed. Source timestamps are timezone-naive; no timezone is assigned to them. All routes use the source clock, and origin-based inference requires a recorded minute. The API runs on `127.0.0.1:8000` by default and has **no authentication**. Do not expose it on a public interface. Deployment and access controls are outside this API implementation.

## Install and start

From the repository root, with the project's existing `.venv`:

```bash
.venv/bin/python -m pip install -r requirements-api.txt
.venv/bin/python scripts/serve_api.py
```

The server loads the same local files used by the command-line scripts. It neither trains models nor creates reports. The OpenAPI schema is served at `http://127.0.0.1:8000/openapi.json`. An HTTP `503` indicates missing data or model artifacts; an HTTP `422` indicates an invalid request or data/model compatibility failure. Errors deliberately omit local filesystem paths.

## Routes

| Method and path | Purpose | Input |
|---|---|---|
| `GET /api/v1/health` | Report available local artifacts; not a model validation check | none |
| `GET /api/v1/metrics` | Recorded minute-by-minute KPI series | `start`, `end` (optional), `limit` (1–720, default 120) |
| `GET /api/v1/metrics/summary` | Observed metric values and preceding 60-minute medians | `as_of` |
| `POST /api/v1/forecasts/latency` | Saved `t+5` processing-time forecast | `{"as_of":"2025-10-25T14:24:00"}` |
| `POST /api/v1/forecasts/kpi` | Saved `t+5` utilization or channel-success forecast | `{"target":"cpu_utilization","as_of":"2025-10-25T14:24:00"}` |
| `POST /api/v1/monitoring/anomaly` | Score an actual monitoring minute | `{"as_of":"2025-10-25T18:01:00"}` |
| `POST /api/v1/logs/classify` | Normalized text cluster and exact-template label | `{"message":"...","log_level":"ERROR"}` |
| `GET /api/v1/logs/recent` | Recent normalized, classified logs | `as_of`, `lookback_minutes` (1–120), `limit` (1–20) |
| `POST /api/v1/knowledge/search` | FAISS runbook or incident-document search | `{"query":"network timeout latency"}` |
| `GET /api/v1/incidents/{incident_id}/triage` | Triage an original historical incident record | supplied incident ID, e.g. `INC000136` |
| `POST /api/v1/investigations` | Run local LangGraph evidence-only investigation | `{"as_of":"2025-10-25T18:01:00","question":"network timeout latency"}` |
| `POST /api/v1/investigations/check-draft` | Validate a locally imported model draft against a freshly assembled investigation; no external request | `{"as_of":"...","question":"...","draft":{...}}` |

All `as_of`, `start`, and `end` values use complete ISO dates and times, such as `2025-10-25T18:01:00`, without offsets or fractional seconds. Lists are bounded. `GET /metrics` returns the **latest** `limit` rows in the requested interval, with `older_matching_rows_omitted` indicating truncation; it never invents missing readings.

### Example

```bash
curl -s http://127.0.0.1:8000/api/v1/metrics?limit=2
curl -s -X POST http://127.0.0.1:8000/api/v1/forecasts/kpi \
  -H 'content-type: application/json' \
  -d '{"target":"cpu_utilization","as_of":"2025-10-25T14:24:00"}'
curl -s -X POST http://127.0.0.1:8000/api/v1/investigations \
  -H 'content-type: application/json' \
  -d '{"as_of":"2025-10-25T18:01:00","question":"network timeout latency"}'
```

### Source and model boundaries

The forecast endpoints use the same saved estimators and timestamp checks as `predict_forecast.py` and `predict_kpi.py`. A prediction is a single value five minutes after its origin—not an alert or a probability of failure. The anomaly endpoint returns a review flag, not an established incident. Log clustering groups wording; known event categories come from the separate pattern catalog. The incident triage endpoint accepts only original incident-report rows and must not be interpreted as a live monitoring-alert classifier.

Log-list responses include normalized message text and original source-row references, but not unmasked transaction, user, or server identifiers. Knowledge results include actual runbook passages and citations; they remain local in this version. The investigation endpoint calls the **evidence-only** LangGraph route. There are no API parameters for remote model selection or external-data permission; unsupported fields are rejected. This does not test or enable model-generated explanations.

## Checks

Run the API tests against the original monitoring CSVs and saved forecast models:

```bash
.venv/bin/python tests/test_api.py
```

The following suite also requires the saved anomaly detector, incident-triage model, log classifier, FAISS index, and LangGraph dependencies from the earlier checkpoints:

```bash
.venv/bin/python tests/test_api_integration.py
```

The first suite covers HTTP request validation, recorded metrics, and saved forecasts. The integration suite exercises the saved detectors, log classifier, FAISS index, and local LangGraph route. Neither measures service throughput, public-endpoint security, or live monitoring.

The optional draft-check endpoint accepts JSON from a consent-gated CLI run and reconstructs citations from source records. It cannot generate a model answer or certify the truth of a claim. See `docs/incident_generation.md`.
