# Capstone scope and demonstration

The project implements a **historical banking-operations observability platform** using the supplied monitoring, transaction, log, incident, and runbook files. It is designed to run locally in Codespaces using FastAPI, Streamlit, and Docker Compose. Original datasets and trained model artifacts remain outside Git because redistribution rights have not been established.

The capstone brief does not specify a submission file format. Use the Git repository and a working demonstration unless the course or reviewer provides separate submission instructions.

## Requirement coverage

| Capstone item | Implemented scope and remaining limitation |
|---|---|
| Data exploration, KPI relationships, and visualizations | Source inspection, correlation reports, notebooks, and historical dashboard charts; not a continuously updating feed |
| Forecasting | Five-minute point forecasts for aggregate transaction processing time, CPU, memory, DB pool use, and NEFT/IMPS/RTGS success rates; the processing-time field is **not** independently measured API latency |
| Incident criticality classification | Historical incident-record triage with Accuracy/F1-related evaluation; source KPI discrepancies limit its use on live monitoring rows |
| Anomaly detection | Saved Isolation Forest review flags on historical metrics; the monitoring data lack per-minute confirmed incident labels to measure detection precision/recall |
| NLP log grouping | TF-IDF/KMeans clustering and a separate exact-template event catalog; clusters are not learned incident labels |
| RAG and LangGraph | Local FAISS retrieval and a source-linked evidence briefing; the optional external model draft still needs a permitted live API call for verification |
| AI explanations | Consent-gated CLI generation with source references and optional import into Streamlit; imported evidence IDs are checked, but causal claims are not independently proven |
| Streamlit and FastAPI | Historical dashboard views and read-only local inference API; no live banking telemetry integration |
| Docker Compose deployment | Two local services using read-only data/model mounts, private ports, health probes, and source-backed smoke tests; not a hardened public service |

## Demonstration route

1. Start from a Codespace containing the original ignored files and fitted model artifacts. `docker compose up --build --detach --wait` starts FastAPI and Streamlit. The containers use host ports 18000 and 18501; keep forwarded ports private.
2. Run `.venv/bin/python scripts/smoke_deployment.py` and show the logged source-backed checks. Then open the Streamlit dashboard and show the five existing tabs.
3. In **Incident replay**, select `2025-10-25 18:01:00`. Explain the anomaly review flag while pointing out that the processing-time value is below its prior-hour median. Inspect the network-timeout log and the referenced runbooks; these do not prove a cause.
4. If the original data are approved for external processing and the model API is available, use the separate consent-gated CLI command in `docs/incident_generation.md`, then import and review the returned JSON in Streamlit. Do not present the draft as an established incident diagnosis.
5. Explain the time split, source mismatches, missing incident labels, source-clock ambiguity, and historical-only monitoring. These constraints determine where the results can and cannot be applied.

## Before handing over the repository

Keep `data/raw/`, `models/`, `reports/`, generated notebook outputs, and credentials out of Git. Check `.gitignore` and review `git status` before committing screenshots or results. Retain separate code and README commits. Record the actual Codespaces test summaries; do not state that external generation was verified unless a permitted model request and the draft import both succeeded. The original brief contains no specified requirement for a slide deck, PDF report, or demo video, although supplementary materials may be useful.
