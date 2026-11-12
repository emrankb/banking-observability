# Incident evidence assembly

An investigation starts with a recorded monitoring minute and a question. The
command reads the actual monitoring and application-log files, scores the minute
with the saved anomaly detector, classifies recent messages with the saved log
model, and searches the existing FAISS runbook index.

The command returns JSON with separate fields for observed metrics, model
assessments, classified logs, and runbook matches. Root cause and resolution
remain unset.

## Time boundary

`--as-of` must be an exact minute in `infrastructure_metrics.csv`, after the
knowledge index cutoff and the anomaly detector's calibration period. The
monitoring snapshot uses this minute and its preceding 60 recorded minutes. The
reference medians exclude the current minute. The log window defaults to the
preceding 30 minutes, including the origin; no later message enters the packet.
Logs are returned most recent first, capped at 20, while the total matching
count is retained.

The data files use timezone-naive timestamps with no documented timezone.
Timestamps from other systems must be reconciled before joining them. A
source-file SHA-256 identifies the file version; CSV row numbers locate the
observed values in that file.

## Models and evidence

The anomaly detector's `review_required` flag has the same limited meaning as
in `docs/detection.md`. It is not a root-cause classification. Log event types
come from the exact normalized-template catalog, not KMeans. An unfamiliar
message stays unreviewed.

Retrieval defaults to runbooks. It returns source paths, line ranges, excerpt
hashes and lexical similarity scores. Use `--include-historical-incidents` to
search the earlier incident snapshots separately. Those snapshots have no
validated cause or resolution labels; some of their KPI readings disagree with
monitoring records. The command does not pass live monitoring data to the
incident-snapshot criticality classifier.

The retrieval index includes undated runbooks. For historical replay they are
current reference material, not proof of what operators knew at the time.

## Run the command

Build and verify the existing models and knowledge index first. In the Codespace,
from the repository root:

```bash
.venv/bin/python scripts/investigate_incident.py \
  --as-of "2025-10-25 18:01:00" \
  --question "network timeout latency"
```

At this timestamp the original log file contains a network-timeout message.
The saved monitoring detector provides its own assessment of the recorded
metrics. A runbook match indicates relevant troubleshooting material only.

To include historical case matches:

```bash
.venv/bin/python scripts/investigate_incident.py \
  --as-of "2025-10-25 18:01:00" \
  --question "network timeout latency" \
  --include-historical-incidents
```

## Verification

`tests/test_incident_evidence.py` checks source rows, the preceding baseline,
log selection and temporal isolation against the supplied CSVs. It runs
without saved models. `tests/test_investigation.py` exercises the real FAISS
index and all saved models; missing artifacts are an integration failure. The
source tests do not establish end-to-end behavior on their own.

This is the evidence packet for the planned LangGraph workflow. The next
layer will need to link its conclusions to source records and leave the cause
unresolved when the available evidence does not support one.
