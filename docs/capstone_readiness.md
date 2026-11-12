# Capstone demonstration

This project brings banking transaction metrics, infrastructure readings, application logs, and incident records into one local application. The dashboard replays the supplied data, while saved models provide five-minute forecasts, anomaly review flags, log grouping, and incident evidence. FastAPI serves the results to Streamlit. Docker Compose runs both services in Codespaces.

## How the parts fit together

- `scripts/` contains the data checks, training commands, report generation, and CLI tools. Training writes to ignored `models/` and `reports/` directories.
- The forecasting models predict the next recorded processing-time reading and six infrastructure or payment-channel readings. The anomaly detector works on recorded monitoring minutes; the separate criticality classifier scores incident-report records.
- TF-IDF/KMeans groups recurring log messages. A local FAISS index searches runbooks and historical incident material. LangGraph brings the monitoring readings, logs, and search results together for an investigation.
- FastAPI serves historical metrics, model results, and evidence checks. Streamlit shows them in five views. An optional, separate CLI request can produce a model-written explanation; the dashboard imports that JSON and checks its cited evidence IDs against the original records.

## Requirements and limits

| Part of the brief | What was built | Important qualification |
|---|---|---|
| Data exploration | Dataset checks, distributions and correlations in reports/notebooks; KPI charts in Streamlit | Correlation does not establish a cause |
| Forecasting | Five-minute point forecasts for processing time, CPU, memory, DB pool use, and NEFT/IMPS/RTGS success rates | The processing-time field is not an API latency measurement |
| Incident classification | Criticality prediction for supplied historical incident records | Differences between incident records and the monitoring CSV restrict its use as a monitoring alert |
| Anomaly detection | Isolation Forest on recorded minute-level metrics | Flags request investigation; there are no confirmed per-minute incident labels for detection accuracy |
| Log analysis | TF-IDF/KMeans clustering and an exact-pattern event catalog | Text similarity is not a verified issue category or root cause |
| Retrieval and incident assistant | FAISS source lookup and LangGraph evidence assembly | Retrieved runbooks are guidance, not proof of a cause |
| Model-generated explanation | An opt-in external-model CLI produces a structured, cited draft; Streamlit checks and displays the imported JSON | Citations are checked against sources, but the wording and diagnosis still need human review |
| Dashboard and API | Streamlit and FastAPI over recorded CSVs | Historical replay, not a live data feed |
| Deployment | Local Docker Compose API and dashboard with private host ports | No authentication or public production deployment |

## Demo in Codespaces

Start with the supplied CSVs in `data/raw/` and the saved artifacts in `models/`. Both directories are ignored by Git. The model files are needed for inference, and the services do not train them on startup.

```bash
docker compose up --build --detach --wait
docker compose ps
.venv/bin/python scripts/smoke_deployment.py
```

Open the **private** Codespaces port `18501`. Show Monitoring, Forecasts, Log explorer, Runbooks, and Incident replay. The API is available locally on port `18000`.

For Incident replay, select `2025-10-25 18:01:00`, run the anomaly score, and assemble the evidence briefing for `network timeout latency`. The detector flags that minute for review; the logs and linked runbooks help explain what to inspect, but they do not confirm an incident cause.

A model-generated JSON draft for that same minute was successfully produced through `scripts/ask_incident.py` with `gpt-5.4-mini` and imported into Streamlit. For the demo, the file is stored locally under `reports/incident_drafts/network-timeout-1801.json`. Use the explanation import control and **Check draft against recorded evidence** to display the summary, possible explanations, suggested checks, and citations. Reusing this saved file does not make another API request. The command and its data-sharing requirements are in `docs/incident_generation.md`.

## Checks completed in Codespaces

- Docker: both containers reached healthy status and the 12 source-backed deployment smoke checks passed.
- Deployment contract: 17 tests passed.
- Incident-generation changes: 52 assistant, draft-validation, and saved-model integration tests passed before the final citation fix was merged. They do not contact the model API.
- A separate live model call returned `model_draft_with_source_links`; the JSON was accepted and displayed by the dashboard. Its `root_cause` and `resolution` fields stayed unset.

These are the observed checks, not a claim that the system has been tested against every incident or question. The demo still depends on the original local files and trained models.

## What remains outside the scope

The source timestamps do not define a timezone, the monitoring history is not a live feed, and the saved forecasts are point estimates rather than incident probabilities. Some incident-report KPIs disagree with the monitoring source at matching times. Model-generated explanations are investigative drafts, not automated remediation or verified fault diagnoses.

Keep `data/raw/`, `models/`, `reports/`, notebook outputs, and API keys out of Git. The source repository alone cannot reproduce a model-backed demo without the separately supplied datasets and trained artifacts. The capstone brief does not prescribe a slide deck, PDF report, or video; check the course submission instructions before handing in the repository.
