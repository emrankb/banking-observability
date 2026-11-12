"""Build traceable passages from supplied runbooks, project notes, and incident rows."""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .detection_common import INCIDENT_METRICS, MONITORING_METRICS, metric_frame, read_incidents
from .forecasting import file_sha256

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("supplied_runbook", "project_runbook", "incident_snapshot")
MAX_PASSAGE_CHARS = 1400


def naive_time(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("Use a non-null timezone-naive source timestamp.")
    return result


def source_ref(path: Path, logical_path: str, start: int, end: int) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not 1 <= start <= end <= len(lines):
        raise ValueError(f"Invalid source span for {logical_path}.")
    excerpt = "\n".join(lines[start - 1:end])
    return {"path": logical_path, "sha256": file_sha256(path), "line_start": start,
            "line_end": end, "excerpt": excerpt, "excerpt_sha256": sha256(excerpt.encode()).hexdigest()}


def make_chunk(document_id: str, title: str, kind: str, text: str, references: list[dict],
               *, timestamp: str | None = None, event_type: str | None = None,
               audit: list[dict] | None = None, verbatim: bool = True) -> dict:
    if kind not in KINDS or not text.strip() or not references:
        raise ValueError("Passages require a known kind, text, and source references.")
    body = {"document_id": document_id, "title": title, "kind": kind, "text": text,
            "references": references, "event_timestamp": timestamp, "event_type": event_type,
            "text_is_verbatim": verbatim, "monitoring_disagreements": audit or []}
    identity = {key: body[key] for key in ("document_id", "text", "references")}
    digest = sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return {"chunk_id": document_id + ":" + digest, **body}


def line_windows(lines: list[str], start: int, stop: int):
    """Yield bounded inclusive line spans; overlap one line only when a split is needed."""
    while start <= stop:
        while start <= stop and not lines[start - 1].strip():
            start += 1
        if start > stop:
            break
        end, size = start - 1, 0
        while end < stop:
            length = len(lines[end]) + 1
            if length > MAX_PASSAGE_CHARS:
                raise ValueError("A source line exceeds the passage limit; split the source line first.")
            if size + length > MAX_PASSAGE_CHARS:
                break
            end += 1
            size += length
        yield start, end
        if end == stop:
            break
        start = end if end > start else end + 1


def runbook_chunks(path: Path, logical_path: str, *, project: bool = False) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = "## " if project else "TITLE: "
    starts = [i for i, line in enumerate(lines, 1) if line.startswith(marker)]
    if not starts:
        raise ValueError(f"No runbook headings in {logical_path}.")
    chunks, doc_ids = [], set()
    for offset, start in enumerate(starts):
        stop = starts[offset + 1] - 1 if offset + 1 < len(starts) else len(lines)
        while stop > start and (not lines[stop - 1].strip() or set(lines[stop - 1]) == {"="}):
            stop -= 1
        title = lines[start - 1].removeprefix(marker).strip()
        event_type = None
        if project:
            events = [line.removeprefix("Event type: ").strip() for line in lines[start:stop]
                      if line.startswith("Event type: ")]
            if len(events) != 1 or not re.fullmatch(r"[a-z][a-z0-9_]+", events[0]):
                raise ValueError(f"Expected one event type for {title}.")
            event_type = events[0]
            document_id = "runbook:" + event_type
        else:
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            document_id = "supplied:" + slug
        if document_id in doc_ids:
            raise ValueError(f"Duplicate document identifier: {document_id}")
        doc_ids.add(document_id)
        # Keep the scope paragraph and each action section as separate passages.
        sections = [start] + [i for i in range(start + 1, stop + 1)
                              if lines[i - 1].startswith("### ") or lines[i - 1] == "Resolution Steps:"]
        for section, a in enumerate(sections):
            b = sections[section + 1] - 1 if section + 1 < len(sections) else stop
            for lo, hi in line_windows(lines, a, b):
                ref = source_ref(path, logical_path, lo, hi)
                chunks.append(make_chunk(document_id, title,
                                         "project_runbook" if project else "supplied_runbook",
                                         ref["excerpt"], [ref], event_type=event_type))
    return chunks


def csv_spans(path: Path, key: str) -> dict[str, tuple[int, int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or key not in reader.fieldnames:
            raise ValueError(f"CSV is missing {key}: {path.name}")
        spans, previous = {}, reader.line_num
        for row in reader:
            value = row[key]
            if not value or value in spans:
                raise ValueError(f"Empty or duplicate {key}: {path.name}")
            spans[value] = (previous + 1, reader.line_num)
            previous = reader.line_num
    return spans


def collect_knowledge(data_dir: Path, *, notes: Path = ROOT / "knowledge/runbooks.md",
                      catalog: Path = ROOT / "config/log_patterns.json",
                      cutoff: str | None = None) -> tuple[list[dict], dict]:
    data_dir, notes, catalog = Path(data_dir), Path(notes), Path(catalog)
    required = ("incident_knowledge_base.txt", "incident_reports.csv", "infrastructure_metrics_train.csv")
    for path in [*(data_dir / name for name in required), notes, catalog]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing knowledge input: {path}")
    train_path = data_dir / "infrastructure_metrics_train.csv"
    monitoring = metric_frame(pd.read_csv(train_path), MONITORING_METRICS, step_minutes=1)
    boundary = naive_time(cutoff) if cutoff is not None else monitoring.timestamp.iloc[-1]
    if not monitoring.timestamp.eq(boundary).any():
        raise ValueError("Knowledge cutoff must match a timestamp in the monitoring training file.")
    incidents_path = data_dir / "incident_reports.csv"
    incidents = read_incidents(incidents_path)
    raw = pd.read_csv(incidents_path)
    for column in ("severity", "incident_type"):
        if column not in raw or raw[column].isna().any():
            raise ValueError(f"Incident source requires non-null {column}.")
        incidents[column] = raw[column].astype(str)
    eligible = incidents.loc[incidents.timestamp.le(boundary)]
    if eligible.empty:
        raise ValueError("No incident records at or before the cutoff.")
    lookup = monitoring.set_index("timestamp")
    if not eligible.timestamp.isin(lookup.index).all():
        raise ValueError("Every included incident must have an exact monitoring timestamp.")
    chunks = runbook_chunks(data_dir / required[0], "data/raw/" + required[0])
    authored = runbook_chunks(notes, "knowledge/runbooks.md", project=True)
    tags = {item["event_type"] for item in authored}
    catalog_items = json.loads(catalog.read_text(encoding="utf-8"))["patterns"]
    expected = {item["event_type"] for item in catalog_items if item["category"] != "routine"}
    if tags != expected:
        raise ValueError(f"Runbook/catalog mismatch: missing={sorted(expected-tags)}, extra={sorted(tags-expected)}")
    chunks.extend(authored)
    incident_spans = csv_spans(incidents_path, "incident_id")
    metric_spans = csv_spans(train_path, "timestamp")
    conflict_rows, conflict_cells = 0, 0
    for _, incident in eligible.iterrows():
        observed = lookup.loc[incident.timestamp]
        differences = [{"metric": col, "incident_value": float(incident[col]),
                        "monitoring_value": float(observed[col])} for col in INCIDENT_METRICS
                       if not np.isclose(incident[col], observed[col], rtol=0, atol=1e-8)]
        conflict_rows += bool(differences)
        conflict_cells += len(differences)
        title = f"{incident.incident_id}: {incident.incident_type} ({incident.severity})"
        text = [title, f"Recorded timestamp: {incident.timestamp.isoformat()}",
                f"Supplied is_critical label: {int(incident.is_critical)}", "Incident-table KPIs:"]
        text.extend(f"{col}: {float(incident[col]):.12g}" for col in INCIDENT_METRICS)
        if differences:
            text.append("The incident and monitoring sources disagree at this timestamp:")
            text.extend(f"{d['metric']}: incident={d['incident_value']:.12g}; monitoring={d['monitoring_value']:.12g}" for d in differences)
        else:
            text.append("The shared KPIs match the monitoring row within the audit tolerance.")
        text.append("No confirmed root cause, resolution, or recovery outcome is supplied for this record.")
        refs = [source_ref(incidents_path, "data/raw/incident_reports.csv", *incident_spans[incident.incident_id]),
                source_ref(train_path, "data/raw/infrastructure_metrics_train.csv", *metric_spans[str(incident.timestamp)])]
        chunks.append(make_chunk("incident:" + incident.incident_id, title, "incident_snapshot", "\n".join(text),
                                 refs, timestamp=incident.timestamp.isoformat(), audit=differences, verbatim=False))
    ids = [row["chunk_id"] for row in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate passage identifiers.")
    source_paths = {"data/raw/" + name: data_dir / name for name in required}
    source_paths.update({"knowledge/runbooks.md": notes, "config/log_patterns.json": catalog})
    summary = {"format_version": 1, "cutoff": boundary.isoformat(), "chunk_count": len(chunks),
               "document_count": len({c["document_id"] for c in chunks}),
               "documents_by_kind": dict(Counter({c["document_id"]: c["kind"] for c in chunks}.values())),
               "chunks_by_kind": dict(Counter(c["kind"] for c in chunks)),
               "included_incident_rows": len(eligible), "excluded_later_incident_rows": len(incidents) - len(eligible),
               "included_incident_conflict_rows": int(conflict_rows), "included_incident_conflict_cells": conflict_cells,
               "runbook_event_types": sorted(tags), "source_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
               "source_timezone": "unspecified; timezone-naive timestamps",
               "runbook_availability": "Publication and effective dates were not supplied; runbooks are current reference material, not time-verified historical knowledge.",
               "incident_scope": "Supplied snapshots and cross-source checks, not confirmed causes or completed resolutions."}
    return chunks, summary


def validate_chunk(chunk: dict) -> None:
    if chunk.get("kind") not in KINDS or not chunk.get("text") or not chunk.get("document_id"):
        raise ValueError("Invalid passage structure.")
    for ref in chunk.get("references", []):
        path = Path(ref["path"])
        if path.is_absolute() or ".." in path.parts or not 1 <= ref["line_start"] <= ref["line_end"]:
            raise ValueError("Invalid citation path or lines.")
        if sha256(ref["excerpt"].encode()).hexdigest() != ref["excerpt_sha256"]:
            raise ValueError("Citation excerpt checksum mismatch.")
    if not chunk.get("references"):
        raise ValueError("A passage must cite a source.")
    rebuilt = make_chunk(chunk["document_id"], chunk["title"], chunk["kind"], chunk["text"], chunk["references"],
                         timestamp=chunk["event_timestamp"], event_type=chunk["event_type"],
                         audit=chunk["monitoring_disagreements"], verbatim=chunk["text_is_verbatim"])
    if rebuilt != chunk:
        raise ValueError("Passage identity or fields are inconsistent.")
    if chunk["text_is_verbatim"] and (len(chunk["references"]) != 1 or chunk["text"] != chunk["references"][0]["excerpt"]):
        raise ValueError("Verbatim passage differs from its citation.")
    if chunk["event_timestamp"] is not None:
        naive_time(chunk["event_timestamp"])


def source_report(summary: dict) -> str:
    lines = ["# Knowledge sources", "", f"Incident cutoff: {summary['cutoff']}.", "",
             "| Source kind | Documents | Passages |", "|---|---:|---:|"]
    for kind in KINDS:
        lines.append(f"| {kind} | {summary['documents_by_kind'].get(kind,0)} | {summary['chunks_by_kind'].get(kind,0)} |")
    lines += ["", f"Included incident rows: {summary['included_incident_rows']}; later rows excluded: {summary['excluded_later_incident_rows']}.",
              f"Included rows with monitoring disagreements: {summary['included_incident_conflict_rows']} ({summary['included_incident_conflict_cells']} cells).",
              "", "Every runbook passage retains its source line span. Incident cards cite both the incident row and its exact-timestamp monitoring row.",
              "Source runbooks are unchanged. Project notes are investigation guidance, not historical resolutions.",
              summary['runbook_availability'], "", "This source inventory does not verify FAISS indexing or retrieval quality.", ""]
    return "\n".join(lines)


def preview_sources(data_dir: Path, output: Path, *, cutoff: str | None = None) -> dict:
    output, data_dir = Path(output).resolve(), Path(data_dir).resolve()
    if output.exists() or output == data_dir or output in data_dir.parents or data_dir in output.parents:
        raise ValueError("Use a new output directory separate from source data.")
    chunks, summary = collect_knowledge(data_dir, cutoff=cutoff)
    for chunk in chunks:
        validate_chunk(chunk)
    output.mkdir(parents=True, exist_ok=False)
    (output / "chunks.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n" for c in chunks))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "source_review.md").write_text(source_report(summary))
    return summary
