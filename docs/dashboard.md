# Historical operations dashboard

The Streamlit dashboard reads the project's existing FastAPI service. It uses
recorded monitoring and application-log data, the saved prediction models, and
the local retrieval index. Neither service downloads input datasets or trains
models when the dashboard starts.

## Run locally

The original CSVs must remain under `data/raw/`, and the forecast, detection,
log-clustering, and knowledge artifacts must already be in `models/`.

```bash
.venv/bin/python -m pip install -r requirements-dashboard.txt
```

Start FastAPI in one terminal:

```bash
.venv/bin/python scripts/serve_api.py
```

Start Streamlit in another:

```bash
.venv/bin/python scripts/serve_dashboard.py
```

Open `http://127.0.0.1:8501` locally. Streamlit and FastAPI both bind to
loopback addresses. Their endpoints have **no authentication**: don't publish
or forward either port to an unauthenticated public URL. The dashboard uses
`http://127.0.0.1:8000` by default; set `BANKING_API_URL` to another private
HTTP origin if the backend runs elsewhere. In Codespaces, use its **private**
forwarded port 8501 to open the dashboard; do not make the port public.
Visitors cannot select an API URL from within the dashboard.

## Recorded-time controls

Choose a date and minute in the sidebar. The API returns up to 720 of the most
recent recorded minutes at or before that time. The **Recorded replay minute**
control moves within that result. All visualized monitoring values are filtered
to the selected minute, and related API calls use it as `as_of`. The source has
no declared timezone; the dashboard does not assign one.

The five views are:

| View | Content |
|---|---|
| Monitoring | Original KPI readings, prior 60-minute medians, and recorded metric trends |
| Forecasts | Saved five-minute processing-time, resource, and payment success-rate point predictions |
| Log explorer | Recent normalized logs, exact-template event labels, and message classification |
| Runbooks | Lexical retrieval from the local FAISS index, with file/line references |
| Incident replay | Anomaly review flags and a cited LangGraph evidence briefing for the selected recorded minute |

**Incident replay is not fault injection.** Move through recorded minutes to
inspect the monitoring readings and log classifications at each point. It does
not alter the CSVs or manufacture failures. At
`2025-10-25 18:01:00`, the anomaly detector requests review and the briefing
includes the observed monitoring, recent logs, and retrieved runbooks.

The assistant route is **evidence-only**. `root_cause`, `resolution`, and
`model_explanation` remain unset. Forecasts are point readings at `t+5`, not
probabilities or prediction intervals. Runbook similarity is lexical and does
not confirm that an incident occurred or establish its cause.

## Verification

```bash
.venv/bin/python tests/test_dashboard.py
.venv/bin/python tests/test_dashboard_integration.py
```

The first suite routes HTTP calls through the real FastAPI app using the
original monitoring CSVs and saved forecast artifacts, and checks replay
boundaries, source values, and responses. The second additionally needs
Streamlit, the saved anomaly and log models, the FAISS knowledge index, and
the local LangGraph workflow. There is no external model call in either suite.

After both suites pass, start the two servers and check the browser views and
`http://127.0.0.1:8000/api/v1/health`. These checks do not measure browser
performance, public-endpoint security, or production data delivery. Docker
Compose and deployment remain separate work.
