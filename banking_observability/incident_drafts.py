"""Check imported incident drafts against an independently assembled evidence packet."""
from __future__ import annotations

import json
from typing import Any

from .incident_assistant import briefing_from_context, check_generated_reply

MAX_DRAFT_BYTES = 65536
_FIELDS = ("observations", "hypotheses", "next_checks")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field in the imported draft.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("Imported JSON cannot contain " + value + ".")


def read_draft_json(contents: bytes) -> dict:
    """Parse a small local JSON export; do not read arbitrary server-side paths."""
    if not isinstance(contents, bytes) or not 0 < len(contents) <= MAX_DRAFT_BYTES:
        raise ValueError("Select a JSON draft no larger than 64 KiB.")
    try:
        draft = json.loads(contents.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_constant=_invalid_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("The imported draft is not UTF-8 JSON.") from exc
    if not isinstance(draft, dict):
        raise ValueError("Expected a JSON object exported by ask_incident.py.")
    return draft


def validate_imported_draft(draft: dict, context: dict) -> dict:
    """Validate all stored citations using the fresh source-linked context.

    This checks consistency, not whether a generated statement is true.
    """
    baseline = briefing_from_context(context)
    if not isinstance(draft, dict) or set(draft) != set(baseline):
        raise ValueError("The draft does not have the incident briefing fields.")
    if draft["status"] != "model_draft_with_source_links":
        raise ValueError("Import a model-generated briefing, not local evidence-only output.")
    for field, expected in baseline.items():
        if field not in {"status", "model_explanation"} and draft[field] != expected:
            raise ValueError(f"The draft's {field} differs from the selected investigation.")

    explanation = draft["model_explanation"]
    if not isinstance(explanation, dict) or set(explanation) != {
        "summary", "observations", "hypotheses", "next_checks", "uncertainty"
    }:
        raise ValueError("The draft does not contain a valid model explanation.")
    # The CLI has already inserted source_references into its checked output.
    # Rebuild them from the current evidence rather than trusting uploaded paths.
    raw = {"summary": explanation["summary"], "uncertainty": explanation["uncertainty"]}
    for field in _FIELDS:
        rows = explanation[field]
        if not isinstance(rows, list):
            raise ValueError("Invalid generated claim list: " + field)
        raw[field] = []
        for entry in rows:
            if not isinstance(entry, dict) or set(entry) != {"text", "evidence_ids", "source_references"}:
                raise ValueError("Imported claims must include their original source references.")
            raw[field].append({"text": entry["text"], "evidence_ids": entry["evidence_ids"]})
    verified = check_generated_reply(raw, context)
    if explanation != verified:
        raise ValueError("An imported source reference differs from the current evidence.")
    return {
        "status": "citation_ids_checked",
        "as_of": context["as_of"],
        "question": context["question"],
        "review_required": context["review_required"],
        "model_explanation": verified,
        "limitation": "Citations match the recorded sources; the claims and suggested checks still require human review.",
    }
