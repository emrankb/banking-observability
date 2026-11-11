#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 - <<'CHECK'
import sys
if sys.version_info[:2] != (3, 13):
    raise SystemExit(f"Python 3.13 is required; found {sys.version.split()[0]}.")
CHECK

python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements-notebook.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_environment.py
mkdir -p data/raw
printf '\nEnvironment ready. No datasets have been downloaded.\n'
