# Banking observability

A project for investigating payment-channel performance, forecasting processing
time, and explaining operational incidents from metrics and application logs.

## Development environment

This branch sets up Python 3.13 and notebook dependencies. The analysis and
forecasting code is being imported separately; this configuration does not
start an API, dashboard, or model-training job.

Open the branch in GitHub Codespaces. Its setup script creates `.venv`, installs
`requirements-notebook.txt`, checks dependency compatibility, and imports each
pinned package. In the terminal, check the environment again with:

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
never downloads or generates data. The supplied files' redistribution rights
have not been established, so raw data and generated reports and models are
excluded from version control. Review notebook outputs before publishing them.

## Verification

`check_environment.py` checks Python, installed package versions, and imports.
It does not run model tests or verify application deployment. Container startup
must be checked in Codespaces before this environment is described as verified.
