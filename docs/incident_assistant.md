# Incident assistant

The LangGraph workflow calls the existing investigation code, which reads the saved monitoring, log, and FAISS artifacts without changing them. It builds a source-linked briefing and can optionally request a model-generated explanation.

## Workflow

`gather` calls `assemble_investigation()`, which reads monitoring history through the chosen minute, classifies recent logs with the saved artifact, and retrieves runbooks from the FAISS index. `reduce` retains ten monitoring values with their prior-hour medians, up to twelve normalized log messages, and up to three retrieved runbook passages. Each item carries an evidence ID and its source location.

In local mode, `reduce` goes straight to `finalize`, returning the monitoring values, log and runbook references, and the detector's review flag. No API request is made.

With a model selected, `generate` asks the OpenAI Responses API for a structured draft, and `finalize` checks its fields and evidence IDs. The response separates observations, possible causes, next checks, and uncertainty. Claims have source links, but an ID match cannot establish the correctness of the claim. Treat model output as an analyst draft that needs review.

## Run locally

Install the new dependencies in the existing Codespace:

```bash
.venv/bin/python -m pip install -r requirements-assistant.txt
```

Use the saved log, anomaly, and knowledge artifacts from the previous checkpoints:

```bash
.venv/bin/python scripts/ask_incident.py \
  --as-of "2025-10-25 18:01:00" \
  --question "network timeout latency"
```

The result includes ten monitoring values with prior-hour comparisons, recent catalog-labeled log patterns, runbook references, and the detector's review flag. It leaves `root_cause` and `resolution` unset. The sample minute has a monitoring review flag, but that is not a verified incident label.

## Optional external generation

Nothing in the local command makes an API request. To generate a model draft, configure `OPENAI_API_KEY` in the Codespace environment (not in a repository file), choose a model supported by your account, and supply **both** `--model` and `--allow-external-data`:

```bash
.venv/bin/python scripts/ask_incident.py \
  --as-of "2025-10-25 18:01:00" \
  --question "network timeout latency" \
  --model YOUR_AVAILABLE_MODEL \
  --allow-external-data
```

The outgoing request includes the investigation question, ten historical monitoring values and prior-hour medians, up to twelve normalized log messages, and up to three retrieved runbook excerpts, with references. Normalization masks the log IDs represented by known templates; it does not constitute a general privacy guarantee. **Use the external mode only if sharing these banking-related materials with the selected API is permitted.** The code sets `store=False`, but data-processing policies are determined by the provider and account configuration. There is no automated remediation or access to live systems.

The model is asked for structured text. The validator rejects unknown evidence IDs, extra schema fields, uncited statements in the arrays, and runbook-sourced items presented as observations. It does not prove that a citation supports the exact wording of a claim. The final object always retains `root_cause: null` and `resolution: null`; a model draft cannot promote a possible cause to a verified finding.

## Verification

```bash
.venv/bin/python tests/test_assistant_context.py --data-dir data/raw
.venv/bin/python tests/test_assistant_workflow.py --data-dir data/raw
```

The first suite uses the original monitoring CSV to check the response schema, source references, and external-data permission. The second requires LangGraph and the saved models and FAISS index in Codespaces. Neither suite makes an external API request. Performance on unfamiliar questions and model-generated explanations remains unmeasured.

This is a historical analysis command, not a deployed incident-response agent. See `docs/incident_evidence.md` for the data alignment and model-transfer limitations.
