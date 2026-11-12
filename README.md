# Banking observability

A project for investigating payment-channel performance, forecasting processing
time, and explaining operational incidents from metrics and application logs.

The repository currently includes dataset inspection and relationship checks.
Forecasting code is being imported separately. There is no API or dashboard yet.

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

## Tests

```bash
.venv/bin/python tests/test_discovery.py --data-dir data/raw
```

The tests run both scripts, check the supplied dataset snapshot, recompute
correlations and alignment results, verify source files are unchanged, and
compare two independent runs. No saved reports are required. Expected row
counts and anomaly counts describe the supplied files; changed inputs require
reviewing those expectations rather than altering the data to pass the tests.

The environment checker verifies package versions and imports only. Neither
it nor the discovery tests verify a model or application deployment.
