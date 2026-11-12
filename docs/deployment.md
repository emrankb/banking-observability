# Local Docker Compose deployment

The two-service Compose setup runs the historical monitoring API and Streamlit dashboard from one Python image. The API reads the original CSVs and saved models; the dashboard calls the API over the private Compose network. There is no live banking feed, scheduled model training, or external-model generation.

## Before starting

Use Docker Engine with the Compose v2 plugin. Check the installation with `docker compose version`, then build the image and run the HTTP smoke test in Codespaces.

The original CSVs must remain under the ignored `data/raw/` directory, including `infrastructure_metrics.csv`, `infrastructure_metrics_train.csv`, `application_logs.csv`, `incident_reports.csv`, and `incident_knowledge_base.txt`. The saved artifacts from the earlier checkpoints must be under the ignored `models/` directory. These files are **not** downloaded, fabricated, built into the image, or committed to Git. Missing mounts stop Compose before it starts a service; missing files and models leave the API unhealthy.

The image is based on `python:3.13.18-slim-bookworm`, matching the Python patch version recorded in the FAISS index built in Codespaces. It installs the existing pinned application requirements and Debian `libgomp1` for the native scientific libraries. The base-image tag is version-specific but not digest-locked, and indirect Python dependencies are not fully locked; rebuilds are not guaranteed byte-for-byte identical. Saved artifacts still check their recorded feature-code and runtime versions. If a compatibility check fails, inspect the environment rather than bypassing it.

## Build and start

From the repository root:

```bash
docker compose config --quiet
.venv/bin/python tests/test_deployment.py
docker compose up --build --detach --wait
docker compose ps
```

The services listen internally on ports 8000 and 8501. Host access is bound to **127.0.0.1:18000** for FastAPI and **127.0.0.1:18501** for Streamlit. These separate host ports allow the earlier local development servers on 8000 and 8501 to remain running without collisions. Within Compose, Streamlit calls `http://api:8000`.

In Codespaces, open **port 18501** using its **Private** port-forward setting. Do not make the port public. The API has no authentication and neither service is suitable for public-network deployment as configured. Container binding to `0.0.0.0` is required for communication *inside Docker*; published host bindings remain restricted to loopback.

The services run without root privileges, have read-only container filesystems, and mount `data/raw/` and `models/` read-only. A temporary filesystem supplies writable scratch space. The Docker build context uses an allowlist, excluding datasets, model artifacts, reports, credentials, and notebook outputs.

## Verify both services

Use the saved-model health route and the source-backed smoke test:

```bash
curl --fail http://127.0.0.1:18000/api/v1/health
curl --fail http://127.0.0.1:18501/_stcore/health
.venv/bin/python scripts/smoke_deployment.py
```

The smoke test checks HTTP responses for the real October 25, 2025, 18:01 monitoring minute, source-row values and hashes, all seven saved forecasts, anomaly scoring, classified logs, FAISS citations, incident triage, and the local evidence-only LangGraph route. It checks Streamlit's HTTP health endpoint but **does not** prove that all dashboard widgets render. Open the five tabs in a browser and run the incident replay there as a separate visual check.

The test leaves original CSVs and model artifacts unchanged and does not call an external language model. Its default URLs only allow localhost. Run with `--api-url` and `--dashboard-url` if you change the published host ports, for example:

```bash
BANKING_API_PORT=18080 BANKING_DASHBOARD_PORT=18580 docker compose up --build --detach --wait
.venv/bin/python scripts/smoke_deployment.py \
  --api-url http://127.0.0.1:18080 \
  --dashboard-url http://127.0.0.1:18580
```

## Inspect and stop

If the build fails, check the Docker Engine and dependency-install output. If a service is unhealthy, inspect `docker compose ps` and `docker compose logs --tail=100 api dashboard`. The container runs as UID `10001`. In Codespaces, directories created by the `vscode` user can prevent that UID from traversing the bind-mounted model files, even when the files themselves are readable. If the API logs show `PermissionError` under `/app/models/` or `/app/data/raw/`, inspect the host directory permissions with:

```bash
namei -l models/detection/monitoring_anomaly/metadata.json
```

On a dedicated Codespace, grant the container UID read access and directory traversal using ACLs:

```bash
sudo apt-get update && sudo apt-get install -y acl
setfacl -R -m u:10001:rX models data/raw
docker compose up --detach --wait
```

This does not make the mounted files writable to the container. It changes permissions on the host files, so review access requirements before applying it on a shared machine. Do not use `chmod -R 777`. Check model-code, package-version, and source checksums separately if permission errors are resolved but an endpoint still fails. Health checks do not retrain or repair anything.

To stop and remove the two containers:

```bash
docker compose down
```

The source data and saved models are bind-mounted from the workspace, so they remain in their original directories. Do not use `docker compose down --volumes` as a substitute for reviewing data retention. If source files or model artifacts are replaced, repeat the build/source checks and restart the services before rerunning the smoke test.
