# Banking observability

A project for investigating payment-channel performance, forecasting processing
time, and explaining operational incidents from metrics and application logs.

The repository includes dataset inspection, relationship checks, and a
five-minute processing-time forecasting experiment. There is no API or dashboard yet.

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
