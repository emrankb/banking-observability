"""Local Streamlit view of recorded banking operations and model outputs."""
from __future__ import annotations

from datetime import date, time

import pandas as pd
import streamlit as st

from banking_observability.dashboard import (
    DashboardAPI, DashboardError, FORECAST_TARGETS, KPI_GROUPS,
    evidence_links, forecast_fields, monitoring_frame, recorded_minutes,
    source_date_time, summary_by_metric,
)


@st.cache_resource(show_spinner=False)
def api_client() -> DashboardAPI:
    return DashboardAPI()


def query(operation):
    try:
        return operation()
    except (DashboardError, ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))
        return None


def format_metric(value: float, label: str) -> str:
    suffix = " ms" if label == "avg_transaction_processing_time_ms" else "%" if (
        label.endswith("_rate") or label in {
            "cpu_utilization", "memory_usage", "db_connection_pool_usage", "service_uptime"}) else ""
    return f"{float(value):,.2f}{suffix}"


def show_overview(api: DashboardAPI, as_of: str, frame: pd.DataFrame) -> None:
    st.subheader("Monitoring overview")
    st.caption("Recorded measurements at the replay minute; comparisons use the preceding 60 minutes.")
    summary = query(lambda: api.summary(as_of=as_of))
    if summary:
        by_name = summary_by_metric(summary)
        selected = ["avg_transaction_processing_time_ms", "application_error_rate",
                    "neft_success_rate", "cpu_utilization"]
        columns = st.columns(4)
        for column, name in zip(columns, selected):
            if name not in by_name:
                continue
            row = by_name[name]
            with column:
                st.metric(name.replace("_", " ").title(), format_metric(row["observed"], name),
                          delta=f"{float(row['difference_from_median']):+.2f} vs median", delta_color="off")
    group = st.selectbox("Metric group", list(KPI_GROUPS), key="metric_group")
    fields = [name for name in KPI_GROUPS[group] if name in frame.columns]
    chosen = st.multiselect("Plot recorded series", fields, default=fields[:3], key=f"chart_fields_{group}")
    if chosen:
        st.line_chart(frame[chosen], height=340, x_label="Source minute")
        st.caption("No interpolation, future readings, or live updates. Units are shown in the metric-group label.")
    with st.expander("Source readings", expanded=False):
        st.dataframe(frame.tail(30), use_container_width=True)


def show_forecasts(api: DashboardAPI, as_of: str, frame: pd.DataFrame) -> None:
    st.subheader("Five-minute forecasts")
    st.caption("Predictions are issued at the selected historical minute for the observation five minutes later.")
    with st.form("forecast_request"):
        label = st.selectbox("Forecast target", list(FORECAST_TARGETS))
        requested = st.form_submit_button("Run saved forecast", type="primary")
    if requested:
        reply = query(lambda: api.forecast(as_of=as_of, target=FORECAST_TARGETS[label]))
        st.session_state["forecast_reply"] = reply
        st.session_state["forecast_origin"] = as_of
        st.session_state["forecast_target"] = FORECAST_TARGETS[label]
    reply = (st.session_state.get("forecast_reply")
             if st.session_state.get("forecast_origin") == as_of
             and st.session_state.get("forecast_target") == FORECAST_TARGETS[label] else None)
    if reply:
        result = query(lambda: forecast_fields(reply))
        if result:
            x, y = st.columns(2)
            x.metric("Forecast at " + result["target_time"].replace("T", " "),
                     f"{result['value']:.2f} {result['unit']}")
            y.metric("Selected candidate", str(result["model"]))
            st.caption("Point estimate, not a probability or prediction interval. The future actual is not used to produce this value.")
            if result["target"] in frame:
                st.line_chart(frame[[result["target"]]].tail(120), height=280, x_label="Recorded minute")
    else:
        st.info("Choose a target and run the saved model. No new training is performed.")


def show_logs(api: DashboardAPI, as_of: str) -> None:
    st.subheader("Recent log messages")
    lookback = st.slider("Lookback (minutes)", min_value=1, max_value=120, value=30)
    data = query(lambda: api.logs(as_of=as_of, lookback_minutes=lookback))
    if data:
        rows = data.get("rows", [])
        st.caption(f"{len(rows)} of {data.get('matching_source_rows', 0)} matching source records. The API limits results to 20.")
        if rows:
            cols = ["timestamp", "csv_row", "log_level", "event_type", "category",
                    "normalized_message", "label_review_required"]
            st.dataframe(pd.DataFrame(rows)[cols], hide_index=True, use_container_width=True)
        else:
            st.info("There are no log records in this interval.")
    st.divider()
    st.subheader("Classify a message")
    with st.form("log_classification"):
        message = st.text_area("Message text", placeholder="Paste an application log line", max_chars=4096)
        level = st.selectbox("Structured log level", ["ERROR", "WARNING", "CRITICAL", "INFO", "DEBUG"])
        submitted = st.form_submit_button("Classify")
    if submitted:
        result = query(lambda: api.classify(message=message, level=level))
        if result:
            st.write("**Catalog event:**", result.get("event_type") or "unreviewed")
            st.write("**Text cluster:**", result.get("cluster_label") or "none")
            st.write("**Input-quality flags:**", ", ".join(result.get("quality_flags", [])) or "none")
            st.caption("Clusters group wording; catalog labels are separate from incident and cause verification.")


def show_runbooks(api: DashboardAPI, as_of: str) -> None:
    st.subheader("Runbook search")
    st.caption("TF-IDF / FAISS lexical matches. Similarity scores are not confidence or evidence of causation.")
    with st.form("knowledge_lookup"):
        question = st.text_input("Search runbooks", value="network timeout latency", max_chars=1000)
        submitted = st.form_submit_button("Search")
    if submitted:
        st.session_state["knowledge_result"] = query(lambda: api.knowledge(query=question, as_of=as_of))
        st.session_state["knowledge_origin"] = as_of
    result = st.session_state.get("knowledge_result") if st.session_state.get("knowledge_origin") == as_of else None
    if result:
        hits = result.get("results", [])
        if not hits:
            st.info("No source passages matched that query at the current threshold.")
        for hit in hits:
            with st.expander(f"{hit['title']} · similarity {float(hit['score']):.3f}", expanded=False):
                st.text(hit.get("text", ""))
                for ref in hit.get("references", []):
                    st.caption(f"{ref['path']} · lines {ref['line_start']}–{ref['line_end']}")


def show_incident_replay(api: DashboardAPI, as_of: str) -> None:
    st.subheader("Incident replay simulator")
    st.caption("Move the replay minute in the sidebar to inspect actual recorded conditions. No incidents or measurements are injected.")
    st.write("**Replay minute:**", as_of.replace("T", " "))
    left, right = st.columns([1, 1])
    with left:
        anomaly_clicked = st.button("Score this minute", use_container_width=True)
    if anomaly_clicked:
        st.session_state["anomaly_result"] = query(lambda: api.anomaly(as_of=as_of))
        st.session_state["anomaly_origin"] = as_of
    result = st.session_state.get("anomaly_result") if st.session_state.get("anomaly_origin") == as_of else None
    if result:
        with right:
            if result.get("review_required"):
                st.warning("Monitoring review flag raised")
            else:
                st.info("No monitoring review flag at this minute")
        st.write(f"Anomaly score: **{float(result['anomaly_score']):.4f}** · threshold: **{float(result['threshold']):.4f}**")
        st.caption("This is a statistical review flag, not a confirmed incident or a root-cause assessment.")
    st.divider()
    with st.form("investigation_request"):
        question = st.text_input("Investigation question", value="network timeout latency", max_chars=1000)
        include_historical = st.checkbox("Include earlier incident records")
        requested = st.form_submit_button("Assemble evidence briefing", type="primary")
    if requested:
        st.session_state["investigation_result"] = query(lambda: api.investigate(
            question=question, as_of=as_of, include_historical=include_historical))
        st.session_state["investigation_origin"] = as_of
    result = st.session_state.get("investigation_result") if st.session_state.get("investigation_origin") == as_of else None
    if result:
        st.write("**Mode:**", result.get("status"))
        if result.get("review_required"):
            st.warning("Monitoring minute flagged for review")
        summary = st.columns(3)
        for col, name, label in zip(summary,
                                     ("monitoring_metrics", "recent_classified_logs", "runbook_passages"),
                                     ("Observed metrics", "Classified logs", "Runbook passages")):
            col.metric(label, str(result.get(name, 0)))
        refs = evidence_links(result)
        if refs:
            st.dataframe(pd.DataFrame(refs), hide_index=True, use_container_width=True)
        st.info("The briefing does not establish a root cause or resolution. Retrieved procedures are reference material, not authorized actions.")
        if result.get("model_explanation") is not None:
            st.error("Unexpected model output in local evidence-only mode.")


def main() -> None:
    st.set_page_config(page_title="Banking operations | Historical replay", page_icon="📈",
                       layout="wide", initial_sidebar_state="expanded")
    st.title("Banking operations")
    st.caption("Historical observability · Local models · Source clock timezone not specified")
    api = api_client()
    health = query(api.health)
    if not health:
        st.stop()
    if health.get("status") != "ok" or health.get("mode") != "historical_replay":
        st.error("The API is not in historical replay mode.")
        st.stop()
    if health.get("external_generation") is not False:
        st.error("The API unexpectedly allows external generation; dashboard halted.")
        st.stop()
    with st.sidebar:
        st.header("Replay controls")
        st.caption("Choose a source date and minute, then move through recorded readings below.")
        day = st.date_input("Date", value=date(2025, 10, 25))
        clock = st.time_input("Time", value=time(18, 1), step=60)
        end = source_date_time(day, clock)
        requested_minutes = st.slider("History window", 60, 720, 240, step=60)
        st.caption("Source clock is timezone-naive. Dates represent the dataset, not the current time.")
    window = query(lambda: api.metrics(end=end, limit=requested_minutes))
    if not window:
        st.stop()
    options = recorded_minutes(window)
    if not options:
        st.warning("No recorded minutes in that interval.")
        st.stop()
    if st.session_state.get("window_end") != end or st.session_state.get("replay_minute") not in options:
        st.session_state["replay_minute"] = options[-1]
        st.session_state["window_end"] = end
    with st.sidebar:
        replay = st.select_slider("Recorded replay minute", options=options, key="replay_minute")
        st.caption(f"Source minute: {replay.replace('T', ' ')}")
        if not all(health.get("artifacts_present", {}).get(name) for name in
                   ("latency_forecast", "log_classifier", "knowledge_index")):
            st.warning("Some model artifacts are unavailable. Relevant tabs may return an error.")
    frame = monitoring_frame(window)
    frame = frame.loc[frame.index <= pd.Timestamp(replay)]
    tabs = st.tabs(["Monitoring", "Forecasts", "Log explorer", "Runbooks", "Incident replay"])
    with tabs[0]:
        show_overview(api, replay, frame)
    with tabs[1]:
        show_forecasts(api, replay, frame)
    with tabs[2]:
        show_logs(api, replay)
    with tabs[3]:
        show_runbooks(api, replay)
    with tabs[4]:
        show_incident_replay(api, replay)
    st.divider()
    st.caption("Historical replay only · No public authentication · External model generation disabled")


if __name__ == "__main__":
    main()
