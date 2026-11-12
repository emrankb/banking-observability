# Banking observability

A project for investigating payment-channel performance, forecasting processing
time, and explaining operational incidents from metrics and application logs.

The repository includes dataset inspection, relationship checks, five-minute
processing-time and KPI forecasts, incident triage, anomaly detection, log
grouping, local knowledge retrieval, and incident evidence assembly.
The API serves local historical replay. There is no live feed, Streamlit
dashboard, or deployed service yet.

## Development environment

Open the branch in GitHub Codespaces. Its setup script creates a Python 3.13
virtual environment in `.venv`, installs `requirements-notebook.txt`, checks
package compatibility, and imports each pinned package. Check the environment
again with:

```bash
.venv/bin/python scripts/check_environment.py
```

For local development with Python 3.13 and Bash, run the same setup script:

```bash
bash .devcontainer/setup.sh
```

The direct package versions match the existing forecasting checkpoint. They are
not a complete transitive dependency lock. The container tag fixes the Python
minor version and Debian release, but not an immutable image digest.

## Data

Keep the original datasets in `data/raw/`. Setup creates this directory but
never downloads or generates data. Raw data, generated reports, and models are
excluded from version control because redistribution rights have not been
established. Review notebook outputs before publishing them.

The inspection and analysis scripts require these three files:

```text
data/raw/
  transactions_fraud_train.csv
  infrastructure_metrics.csv
  application_logs.csv
```

Add `incident_reports.csv`, `infrastructure_metrics_train.csv`, and
`infrastructure_metrics_test.csv` to the same directory for the incident and
train/test-split checks. The analysis reports which supporting files were not
supplied. The regression tests require all six original files.

## Data discovery

Run these commands from the repository root:

```bash
.venv/bin/python scripts/inspect_data.py --data-dir data/raw --output-dir reports/discovery
.venv/bin/python scripts/analyze_discovery.py --data-dir data/raw --output-dir reports/discovery
```

Open `reports/discovery/discovery_report.md` for the findings. `schema.json`
records column names, types, missing values, categorical counts, and timestamp
coverage. `correlations.csv` compares infrastructure load with channel
non-success percentages and aggregate processing time at three time intervals.
The analysis also checks transaction timestamp overlap, log message patterns,
resource readings, and discrepancies between incident and monitoring records.

`is_fraud` is not an operational failure flag. Channel non-success is calculated
separately as `100 - success_rate` from the infrastructure metrics. Transaction
records are joined to monitoring data by flooring their timestamps to a minute;
only overlapping records are used, assuming a shared source clock. Correlations
are descriptive and do not establish a root cause.

These commands preserve the original CSVs and do not train a model. Some output
files contain individual records, so leave `reports/` outside version control.

## Forecasting

The forecast target is the single `avg_transaction_processing_time_ms` reading
five minutes after the origin. It is not a five-minute average or endpoint
latency. The comparison uses historical lags, trailing-window features, and known
calendar inputs. Models are selected using expanding chronological validation,
then compared on the supplied test period without refitting.

The training and test infrastructure CSVs must be in `data/raw/`. Train locally
in the workspace and generate the report:

```bash
.venv/bin/python scripts/train_forecast.py --data-dir data/raw
.venv/bin/python scripts/write_forecast_report.py --data-dir data/raw
```

An existing checkpoint is not replaced unless `--overwrite` is passed to the
training command. Reports go to `reports/forecast/`; the selected artifact and
learned comparators go to `models/latency_5min/`. They remain outside Git.

Open `reports/forecast/forecast_review.md` for the scores and
`notebooks/02_latency_forecasting.ipynb` for the walkthrough. Select the `.venv`
kernel and run all cells after training; committed notebook outputs are empty.
`docs/forecasting.md` explains the target, folds, and modeling decisions.

Load the saved artifact for a historical replay origin:

```bash
.venv/bin/python scripts/predict_forecast.py --history data/raw/infrastructure_metrics.csv --as-of "2025-10-25 14:24:00"
```

This predicts the 14:29 reading using only history through 14:24. Inference
rejects an origin at or before the artifact's training-target cutoff. The shared
interface requires 1,441 consecutive minute readings. Only load trusted joblib
artifacts; their checksums do not make untrusted deserialization safe.

Baselines are eligible to win and are identified as baselines, not learned
estimators. Validation ranking determines the serving candidate. Discovery
already inspected the full monitoring interval, so the test split is not an
unseen discovery holdout. These are point forecasts, not calibrated prediction
intervals or evidence of future-incident detection.

## Tests

```bash
.venv/bin/python tests/test_discovery.py --data-dir data/raw
```

The tests run both scripts, check the supplied dataset snapshot, recompute
correlations and alignment results, verify source files are unchanged, and
compare two independent runs. No saved reports are required. Expected row
counts and anomaly counts describe the supplied files; changed inputs require
reviewing those expectations rather than altering the data to pass the tests.

After training, run the forecasting suite:

```bash
.venv/bin/python tests/test_forecasting.py --data-dir data/raw
```

It checks causal feature construction, label boundaries, saved-model loading,
command-line inference, and score reproduction. It retrains the full experiment
in a temporary directory and compares the resulting predictions. Run the notebook
separately; the test suite checks its source and output hygiene, not cell execution.

The environment checker verifies package versions and imports only. Neither
these scripts nor the notebooks verify application deployment.

## Incident triage and anomaly detection

Train both detectors from the original incident reports and monitoring train/test
files already in `data/raw/`:

```bash
.venv/bin/python scripts/train_detection.py --data-dir data/raw
```

Artifacts are saved under `models/detection/` and the report is written to
`reports/detection/detection_review.md`. Existing output directories are not
overwritten; use `--model-dir` and `--report-dir` with new paths for another run.
The `--review-fraction` option controls the calibration-tail review budget for
monitoring anomalies; its default is 0.01, not an assumed failure prevalence.

Run the data-dependent checks:

```bash
.venv/bin/python tests/test_detection.py --data-dir data/raw
```

The suite trains temporary runs and checks both saved detectors, temporal
features, threshold calculation, report values, and invalid inputs. It does not
modify your saved models or the original CSVs.

Inspect an incident record or a monitoring minute:

```bash
.venv/bin/python scripts/score_incident.py --incident-id INC000136
.venv/bin/python scripts/score_anomalies.py \
  --history data/raw/infrastructure_metrics.csv \
  --as-of "2025-10-25 18:01:00"
```

Incident triage uses the supplied snapshot labels, whose KPI discrepancies limit
transfer to monitoring alerts. An anomaly flag requests review; it is not a
confirmed failure or root cause. See `docs/detection.md` and
`notebooks/03_incident_detection.ipynb` for the comparisons.

## Log clustering

Train on the earlier messages in `application_logs.csv`, using the monitoring
training cutoff:

```bash
.venv/bin/python scripts/train_logs.py --data-dir data/raw
.venv/bin/python tests/test_logs.py --data-dir data/raw
```

The comparison uses TF-IDF and KMeans on distinct normalized patterns, with
several cluster counts and seeds. Reports go to `reports/logs/`; the saved
vectorizer, centroids, and template catalog go to `models/log_clustering/`.
Existing runs are not overwritten. Use `--model-dir` and `--report-dir` with
new paths for another run.

Read `reports/logs/log_review.md` for group contents and examples. The notebook
`notebooks/04_log_clustering.ipynb` recalculates scores and displays the messages
in each group. Select the `.venv` kernel and run all cells after training.
Save executed copies outside Git.

Inspect an original CSV record with the saved artifact:

```bash
.venv/bin/python scripts/classify_log.py --csv-row 3
```

Row numbers include the CSV header as row 1. The response keeps text grouping
separate from the event type and category assigned by `config/log_patterns.json`.
Unseen patterns stay `unreviewed`; similarity to a known message is not enough
to assign its label. `label_review_required` concerns wording and input quality,
not whether a monitoring incident is occurring. See `docs/logs.md` for details.

## Knowledge retrieval

Add the original `incident_knowledge_base.txt` to `data/raw/`. The index also uses
`incident_reports.csv` and `infrastructure_metrics_train.csv`, plus the project
investigation notes in `knowledge/runbooks.md`.

Install the retrieval dependency in an existing workspace:

```bash
.venv/bin/python -m pip install -r requirements-retrieval.txt
```

New Codespaces install it through `requirements-notebook.txt`. The added package
is `faiss-cpu==1.13.2`; existing scientific-package versions are unchanged.

Check the sources and build the index:

```bash
.venv/bin/python scripts/inspect_knowledge.py --data-dir data/raw
.venv/bin/python scripts/build_knowledge.py --data-dir data/raw
.venv/bin/python scripts/search_knowledge.py --query "NEFT transaction gateway timeout"
.venv/bin/python scripts/evaluate_retrieval.py --data-dir data/raw
```

Source inspection writes `reports/knowledge_sources/`. The FAISS index and encoder
are saved in `models/knowledge/`; query-check results go to `reports/retrieval/`.
Existing directories are not overwritten. Use new `--output-dir` or `--index-dir`
paths for a new run. These files remain outside Git.

Search returns source passages and citations, not a generated answer. Vectors are
local TF-IDF features, not neural embeddings. The default search covers runbooks;
use `--source-kind incident_snapshot` to search incident cards. Those cards retain
conflicting KPIs and do not supply missing causes or resolutions.

Run both test suites:

```bash
.venv/bin/python tests/test_knowledge_sources.py --data-dir data/raw
.venv/bin/python tests/test_retrieval.py --data-dir data/raw
```

The source suite does not require FAISS. The retrieval suite requires the actual
native index and fails when the package is unavailable. Run
`notebooks/05_knowledge_retrieval.ipynb` after building the index to inspect results
and source references. See `docs/retrieval.md` for the cutoff and search limits.

## Incident evidence

Inspect a recorded monitoring minute alongside recent logs, the saved anomaly
assessment, and relevant passages from the local knowledge index:

```bash
.venv/bin/python scripts/investigate_incident.py \
  --as-of "2025-10-25 18:01:00" \
  --question "network timeout latency"
```

The command reads existing artifacts from `models/` and source files from
`data/raw/`. It does not retrain anything or write a report. The response keeps
observed KPI values, classified log messages, monitoring review flags, and
runbook passages separate. It makes no root-cause claim.

The first test suite needs only the original CSVs. The second loads the saved
log, anomaly, and FAISS artifacts already trained in this Codespace:

```bash
.venv/bin/python tests/test_incident_evidence.py --data-dir data/raw
.venv/bin/python tests/test_investigation.py --data-dir data/raw
```

See `docs/incident_evidence.md` and `notebooks/06_incident_evidence.ipynb`.
The language-model interface and dashboard are not part of this checkpoint.

## Incident assistant

The LangGraph workflow uses saved monitoring, log, and FAISS artifacts to create a
source-linked investigation briefing. Install the additional dependencies:

```bash
.venv/bin/python -m pip install -r requirements-assistant.txt
```

Run the local evidence-only route without an API key:

```bash
.venv/bin/python scripts/ask_incident.py \
  --as-of "2025-10-25 18:01:00" \
  --question "network timeout latency"
```

The optional model route requires an API key, a model ID, and an explicit
`--allow-external-data` flag. That route sends a bounded set of monitoring
readings, normalized logs, and retrieved runbook text to the API. Review data
permissions before enabling it. A cited model draft is not a verified diagnosis.

```bash
.venv/bin/python tests/test_assistant_context.py --data-dir data/raw
.venv/bin/python tests/test_assistant_workflow.py --data-dir data/raw
```

The second suite needs the saved artifacts in Codespaces; neither suite makes
an external model call. See `docs/incident_assistant.md` and
`notebooks/07_incident_assistant.ipynb`.

## Additional five-minute KPI forecasts

The processing-time forecast is complemented by direct `t+5` forecasts for CPU,
memory, database connection-pool usage, and NEFT, IMPS, and RTGS success rates.
Fit and test them using the existing monitoring train/test files:

```bash
.venv/bin/python scripts/train_kpi_forecasts.py --data-dir data/raw
.venv/bin/python tests/test_kpi_forecasting.py --data-dir data/raw
```

Replay an origin through the saved model:

```bash
.venv/bin/python scripts/predict_kpi.py \
  --target cpu_utilization \
  --history data/raw/infrastructure_metrics.csv \
  --as-of "2025-10-25 14:24:00"
```

Six selected artifacts go under `models/kpi_5min/`, with comparisons in
`reports/kpi_forecast/`. Existing outputs are not overwritten. The results are
point readings with MAE in percentage points, not calibrated risk forecasts.
See `docs/kpi_forecasting.md` and `notebooks/08_kpi_forecasting.ipynb`.

## Historical monitoring API

The FastAPI service exposes recorded KPI series, saved forecasting and monitoring
inference, normalized log labels, local knowledge search, and evidence-only
investigations. It uses the same ignored data and model artifacts as the CLI
scripts. This is not a live monitoring service.

```bash
.venv/bin/python -m pip install -r requirements-api.txt
.venv/bin/python tests/test_api.py
.venv/bin/python tests/test_api_integration.py
.venv/bin/python scripts/serve_api.py
```

The server binds to `127.0.0.1:8000`. It has no authentication and should not
be exposed to a public network. Check `/api/v1/health` for local artifact
presence, and `/openapi.json` for request/response schemas. Read `docs/api.md`
for endpoint examples and limits. The API does not call an external language
model; remote generation, Streamlit, and Docker are separate checkpoints.
