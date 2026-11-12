"""Fit on earlier log patterns, then inspect assignments on the remaining records."""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from .detection_common import save_detector, write_json
from .forecast_features import read_history
from .forecasting import file_sha256, runtime_versions
from .log_clustering import (NORMALIZER_VERSION, fit_log_clusters, message_evidence,
                             read_catalog, read_logs)


def assign_logs(model, frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cached = {text: model.classify(text, "INFO") for text in frame.normalized_message.unique()}
    rows = []
    for row in frame.itertuples(index=False):
        decision = cached[row.normalized_message]
        evidence = message_evidence(row.log_message, row.log_level)
        if row.log_level.strip().upper() not in model.fit_levels:
            evidence["quality_flags"].append("structured_level_not_seen_during_fit")
        rows.append({"csv_row": row.csv_row, "timestamp": row.timestamp,
                     "split": "development" if row.timestamp <= cutoff else "evaluation",
                     "log_level": row.log_level, "log_message": row.log_message,
                     "normalized_message": row.normalized_message,
                     "duplicate_after_first": row.duplicate_after_first,
                     "cluster_id": decision["cluster_id"], "cluster_label": decision["cluster_label"],
                     "category": decision["category"], "event_type": decision["event_type"],
                     "template_seen_during_fit": decision["template_seen_during_fit"],
                     "label_review_required": not decision["template_seen_during_fit"] or decision["category"] == "unreviewed" or bool(evidence["quality_flags"]),
                     "message_level": evidence["message_level"], "quality_flags": ";".join(evidence["quality_flags"]),
                     **{key: evidence["logged_resource_readings"].get(key, np.nan) for key in ("cpu_pct", "memory_pct")}})
    return pd.DataFrame(rows)


def duplicate_sensitivity(model, development: pd.DataFrame, seed: int) -> dict:
    count = min(1500, len(development))
    indices = np.sort(np.random.default_rng(seed).choice(len(development), count, replace=False))
    sample = development.iloc[indices]
    X = model.vectorizer.transform(sample.normalized_message)
    with threadpool_limits(limits=1):
        labels = model.estimator.predict(X)
    exact_labels = pd.factorize(sample.normalized_message, sort=True)[0]
    return {"sample_rows": count, "sample_seed": seed,
            "sample_templates": sample.normalized_message.nunique(),
            "row_sample_cluster_silhouette": float(silhouette_score(X, labels, metric="cosine")),
            "exact_template_partition_row_silhouette": float(silhouette_score(X, exact_labels, metric="cosine")),
            "meaning": "Repeated identical vectors can inflate row-level silhouette; selection uses one vector per distinct template."}


def report_text(summary: dict, selection: pd.DataFrame, profiles: list[dict]) -> str:
    split, quality = summary["split"], summary["quality"]
    lines = ["# Log clustering", "", "## Data and split", "",
             f"The log file has {summary['rows']:,} records and {summary['raw_unique_messages']:,} distinct raw messages.",
             f"After masking dynamic values there are {summary['normalized_templates']} patterns.",
             f"Fitting uses {split['development_rows']:,} records through {split['cutoff']}; the remaining {split['evaluation_rows']:,} records are evaluated afterward.",
             f"Later records contain {split['evaluation_unseen_templates']} unseen patterns ({split['evaluation_unseen_rows']} rows).",
             "Reappearance of known wording is not a test of new failure modes. Discovery previously inspected the full file.", "",
             "## Choosing k", "",
             "TF-IDF and KMeans use one row per distinct development pattern. Counts are retained for reporting, not fitting.",
             "Every k from 2 through one less than the number of development patterns is compared over three seeds, with 20 initializations per fit.",
             "Selection uses mean cosine silhouette, then mean adjusted Rand agreement between seeds, then smaller k.", "",
             "| k | Mean template silhouette | Minimum silhouette | Seed agreement (ARI) |",
             "|---:|---:|---:|---:|"]
    for row in selection.itertuples(index=False):
        lines.append(f"| {row.k} | {row.mean_template_silhouette:.6f} | {row.min_template_silhouette:.6f} | {row.mean_seed_ari:.6f} |")
    sensitivity = summary["duplicate_sensitivity"]
    lines += ["", f"Selected k: **{summary['selected_k']}**. Final-fit silhouette on distinct patterns: **{summary['final_template_silhouette']:.6f}**.",
              f"On {sensitivity['sample_rows']:,} sampled development records, the same fit scores {sensitivity['row_sample_cluster_silhouette']:.6f}.",
              f"Grouping those sampled records by exact template instead scores {sensitivity['exact_template_partition_row_silhouette']:.6f}.",
              "The higher row-level score reflects repeated wording. It is not evidence of better semantic grouping.", "", "## Group contents", "",
              "Labels below are centroid terms. Category tags come from the explicit template catalog, not from KMeans.",
              "Examples are original development messages; no sample messages were generated.", ""]
    for profile in profiles:
        lines += [f"### Group {profile['cluster_id']}: {profile['label']}", "",
                  f"Development records: {profile['development_rows']}; evaluation records: {profile['evaluation_rows']}.",
                  f"Categories: {', '.join(profile['categories'])}. Distinct fitted patterns: {profile['template_count']}.", ""]
        for example in profile["examples"]:
            lines += [f"- CSV row {example['csv_row']}, `{example['category']}` / `{example['event_type']}`: `{example['message']}`"]
        lines.append("")
    lines += ["## Input quality and inference", "",
              f"Exact duplicate rows after the first: {quality['duplicate_rows']}. They remain in the assignments.",
              f"Structured/message severity disagreements: {quality['severity_mismatches']}.",
              f"Logged CPU readings outside 0–100: {quality['cpu_outside_0_100']}; memory readings outside 0–100: {quality['memory_outside_0_100']}.",
              "The percentages are retained and flagged for measurement-scope review, not clipped or used as clustering features.",
              "Unseen wording receives a tentative cluster when it has known vocabulary, but its category stays unreviewed. Zero-vocabulary input has no cluster.",
              "Cosine similarity is a text-similarity measure, not a confidence probability. A cluster or catalog tag does not verify an incident or root cause.", "",
              "## Files", "",
              "`log_assignments.csv` preserves every input row with its grouping and quality flags. `template_assignments.csv` and `cluster_profiles.json` show all group members.",
              "`selection.json`, `cluster_selection.csv`, and `cluster_seed_results.csv` record the comparison. The notebook reads these files and checks their source hashes.",
              "Generated outputs contain source messages and stay outside version control.", ""]
    return "\n".join(lines)


def run_log_experiment(data_dir: Path, model_dir: Path, report_dir: Path, catalog_path: Path) -> dict:
    data_dir, model_dir, report_dir, catalog_path = (Path(p).resolve() for p in (data_dir, model_dir, report_dir, catalog_path))
    for dest in (model_dir, report_dir):
        if dest == data_dir or dest in data_dir.parents or data_dir in dest.parents or dest == catalog_path or dest in catalog_path.parents:
            raise ValueError("Output directories must be separate from input data and the catalog.")
        if dest.exists():
            raise ValueError(f"Output directory already exists: {dest}. Choose a new path; runs are not overwritten.")
    if model_dir == report_dir or model_dir in report_dir.parents or report_dir in model_dir.parents:
        raise ValueError("Model and report directories must be separate, non-nested paths.")
    for path in (data_dir / "application_logs.csv", data_dir / "infrastructure_metrics_train.csv", catalog_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")
    logs = read_logs(data_dir / "application_logs.csv")
    monitoring = read_history(data_dir / "infrastructure_metrics_train.csv")
    cutoff = monitoring.timestamp.iloc[-1]
    development = logs.loc[logs.timestamp.le(cutoff)]
    evaluation = logs.loc[logs.timestamp.gt(cutoff)]
    if development.empty or evaluation.empty or logs.timestamp.iloc[0] < monitoring.timestamp.iloc[0]:
        raise ValueError("Logs must span the monitoring training cutoff and start inside its observation window.")
    catalog = read_catalog(catalog_path)
    model, plan, tables = fit_log_clusters(development, catalog)
    # The later records do not select k, vocabulary, seeds, or centroid terms.
    assignments = assign_logs(model, logs, cutoff)
    template_rows, profiles = [], []
    for original in model.profiles:
        selected = assignments.loc[assignments.cluster_id.eq(original["cluster_id"])]
        profile = {**original, "development_rows": int(selected.split.eq("development").sum()),
                   "evaluation_rows": int(selected.split.eq("evaluation").sum()), "examples": []}
        for template in original["templates"]:
            members = selected.loc[selected.normalized_message.eq(template)]
            example = members.loc[members.split.eq("development")].iloc[0]
            profile["examples"].append({"csv_row": int(example.csv_row), "message": example.log_message,
                                        "category": example.category, "event_type": example.event_type})
            template_rows.append({"cluster_id": original["cluster_id"], "normalized_message": template,
                                  "category": example.category, "event_type": example.event_type,
                                  "development_rows": int(members.split.eq("development").sum()),
                                  "evaluation_rows": int(members.split.eq("evaluation").sum()),
                                  "example_csv_row": int(example.csv_row), "example_message": example.log_message})
        profiles.append(profile)
    flags = assignments.quality_flags
    unseen = ~evaluation.normalized_message.isin(model.templates)
    source_files = ("application_logs.csv", "infrastructure_metrics_train.csv")
    summary = {"rows": len(logs), "raw_unique_messages": int(logs.log_message.nunique()),
               "normalized_templates": int(logs.normalized_message.nunique()),
               "selected_k": plan["selected_k"], "final_template_silhouette": plan["final_template_silhouette"],
               "source_sha256": {n: file_sha256(data_dir / n) for n in source_files},
               "catalog_sha256": file_sha256(catalog_path), "runtime_versions": runtime_versions(),
               "split": {"cutoff": str(cutoff), "development_rows": len(development), "evaluation_rows": len(evaluation),
                         "development_templates": len(model.templates), "evaluation_templates": int(evaluation.normalized_message.nunique()),
                         "evaluation_unseen_templates": int(evaluation.loc[unseen, "normalized_message"].nunique()),
                         "evaluation_unseen_rows": int(unseen.sum())},
               "quality": {"duplicate_rows": int(logs.duplicate_after_first.sum()),
                           "severity_mismatches": int(flags.str.contains("severity_mismatch").sum()),
                           "cpu_outside_0_100": int(flags.str.contains("cpu_pct_outside").sum()),
                           "memory_outside_0_100": int(flags.str.contains("memory_pct_outside").sum())},
               "category_counts": assignments.category.value_counts().sort_index().to_dict(),
               "mixed_category_clusters": sum(p["mixed_categories"] for p in profiles),
               "duplicate_sensitivity": duplicate_sensitivity(model, development, plan["seed"])}
    with tempfile.TemporaryDirectory(prefix="banking-log-clusters-") as temp:
        scratch = Path(temp); reports = scratch / "reports"; models = scratch / "model"; reports.mkdir()
        save_detector(models, model, {"kind": "log_clustering", "normalizer_version": NORMALIZER_VERSION,
                      "plan": plan, "catalog_sha256": summary["catalog_sha256"], "source_sha256": summary["source_sha256"],
                      "training_code_sha256": file_sha256(Path(__file__))}, module="log_clustering.py")
        for name, table in {**tables, "log_assignments": assignments, "template_assignments": pd.DataFrame(template_rows)}.items():
            table.to_csv(reports / f"{name}.csv", index=False)
        for name, value in (("selection", plan), ("summary", summary), ("cluster_profiles", profiles)):
            write_json(reports / f"{name}.json", value)
        (reports / "log_review.md").write_text(report_text(summary, tables["cluster_selection"], profiles))
        model_dir.parent.mkdir(parents=True, exist_ok=True); report_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(models, model_dir); shutil.copytree(reports, report_dir)
    print(f"LOG CLUSTERS: k={plan['selected_k']}; distinct-pattern silhouette={plan['final_template_silhouette']:.6f}")
    print(tables["cluster_selection"].to_string(index=False))
    print("\nLATER LOGS:", summary["split"])
    print("CATEGORY COUNTS:", summary["category_counts"])
    print("QUALITY FLAGS:", summary["quality"])
    print("Categories come from the template catalog; clusters group similar wording.")
    print("Report:", report_dir / "log_review.md")
    return summary
