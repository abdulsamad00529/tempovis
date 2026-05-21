"""TempoVis — 4-page Streamlit dashboard.

Pages
-----
1. Live Analysis   — upload CSV, call /analyze, show plot + reasoning
2. Alert Feed      — paginated alert table with severity badges
3. Benchmark       — visual vs text comparison from benchmarks/results.json
4. MIMIC Demo      — hardcoded ICU vitals, full agentic pipeline trace
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="TempoVis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _api_get(path: str, **params) -> dict | list | None:
    try:
        resp = httpx.get(f"{API_BASE}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.sidebar.error(f"API error: {exc}")
        return None


def _api_post(path: str, body: dict, timeout: float = 120) -> dict | None:
    try:
        resp = httpx.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API {exc.response.status_code}: {exc.response.text[:300]}")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _severity_badge(sev: str) -> str:
    colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    return f"{colors.get(sev, '⚪')} {sev.upper()}"


# ── Sidebar ────────────────────────────────────────────────────────────────────


with st.sidebar:
    st.title("TempoVis 📈")
    st.caption("Agentic Multimodal Time Series Intelligence")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Live Analysis", "Alert Feed", "Benchmark Results", "MIMIC Demo"],
        index=0,
    )
    st.divider()

    # API status indicator
    with st.container():
        st.caption("API Status")
        health = _api_get("/health")
        if health and health.get("status") == "ok":
            st.success("API online", icon="✅")
        else:
            st.error("API offline", icon="🔴")

    st.divider()
    st.caption(f"Backend: `{API_BASE}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Analysis
# ══════════════════════════════════════════════════════════════════════════════

if page == "Live Analysis":
    st.header("Live Analysis")
    st.caption("Upload a CSV or generate synthetic data, then run the agentic pipeline.")

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        domain = st.selectbox(
            "Domain", ["default", "clinical", "financial", "iot"], index=0
        )
    with col_ctrl2:
        task = st.selectbox(
            "Task",
            ["general", "anomaly_detection", "trend_analysis", "forecasting"],
            index=0,
        )
    with col_ctrl3:
        plot_style = st.selectbox("Plot Style", ["line", "area", "multi_panel"], index=0)

    chain_of_thought = st.toggle("Chain-of-Thought Reasoning", value=True)
    question = st.text_area(
        "Optional focus question",
        placeholder="e.g. Are there anomalies near the end of the series?",
        height=68,
    )

    st.divider()

    tab_upload, tab_manual = st.tabs(["Upload CSV", "Synthetic Generator"])
    series_points: list[dict] = []

    with tab_upload:
        uploaded = st.file_uploader(
            "CSV with `timestamp` + numeric column(s)", type=["csv"]
        )
        if uploaded:
            df = pd.read_csv(uploaded, parse_dates=["timestamp"])
            df = df.sort_values("timestamp")
            val_cols = [c for c in df.columns if c != "timestamp"]
            col = st.selectbox("Value column", val_cols)
            df = df.dropna(subset=[col])
            series_points = [
                {"timestamp": r["timestamp"].isoformat(), "value": float(r[col])}
                for _, r in df.iterrows()
            ]
            st.line_chart(df.set_index("timestamp")[col])
            st.caption(f"{len(series_points)} points loaded")

    with tab_manual:
        n = st.slider("Points", 30, 500, 120, step=10)
        noise = st.slider("Noise", 0.0, 5.0, 1.0, step=0.1)
        anomaly_inject = st.checkbox("Inject anomaly spike", value=True)

        if st.button("Generate"):
            rng = np.random.default_rng(42)
            t = np.linspace(0, 4 * np.pi, n)
            vals = np.sin(t) * 10 + rng.normal(0, noise, n) + np.linspace(0, 5, n)
            if anomaly_inject:
                vals[n // 2] += 30  # spike
            dates = pd.date_range("2024-01-01", periods=n, freq="h")
            series_points = [
                {"timestamp": d.isoformat(), "value": float(v)}
                for d, v in zip(dates, vals)
            ]
            st.session_state["synthetic_points"] = series_points
            st.line_chart(pd.Series(vals, index=dates))
            st.success(f"Generated {n} points")

        if "synthetic_points" in st.session_state and not series_points:
            series_points = st.session_state["synthetic_points"]

    st.divider()

    if not series_points:
        st.info("Add a time series above to enable analysis.")
        st.stop()

    st.markdown(f"**{len(series_points)} points ready** — domain: `{domain}`, task: `{task}`")

    if st.button("Run Analysis", type="primary", use_container_width=True):
        payload = {
            "series": series_points,
            "domain": domain,
            "task": task,
            "plot_style": plot_style,
            "chain_of_thought": chain_of_thought,
            "question": question or None,
        }

        with st.spinner("Running agentic analysis…"):
            data = _api_post("/analyze", payload)

        if not data:
            st.stop()

        st.session_state["last_result"] = data
        result = data.get("result", {})

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence", f"{result.get('confidence', 0):.0%}")
        m2.metric("Latency", f"{data.get('processing_ms', 0)} ms")
        m3.metric("Cost", f"${data.get('cost_usd', 0):.4f}")
        m4.metric("Anomalies", len(result.get("anomalies", [])))

        st.subheader("Summary")
        st.write(result.get("summary", "—"))

        # Plot artifact
        plot_url = data.get("plot_artifact_url")
        if plot_url:
            col_plot, col_json = st.columns([2, 1])
            with col_plot:
                st.subheader("Rendered Plot")
                full_url = (
                    plot_url
                    if plot_url.startswith("http")
                    else f"http://localhost:8000{plot_url}"
                )
                st.image(full_url, use_container_width=True)
        else:
            col_json = st.container()

        with col_json:
            st.subheader("Raw JSON")
            st.json(result, expanded=False)

        # Chain-of-thought
        steps = result.get("reasoning_steps", [])
        if steps:
            st.subheader("Chain-of-Thought Reasoning")
            for step in steps:
                with st.expander(f"Step {step['step']}: {step['observation'][:60]}…"):
                    st.markdown(f"**Observation:** {step['observation']}")
                    st.markdown(f"**Inference:** {step['inference']}")

        # Anomalies highlighted on the input series
        anomalies = result.get("anomalies", [])
        col_a, col_t = st.columns(2)
        with col_a:
            st.subheader("Anomalies")
            if anomalies:
                st.dataframe(pd.DataFrame(anomalies), hide_index=True, use_container_width=True)
                # Overlay anomaly markers
                if series_points:
                    df_plot = pd.DataFrame(series_points)
                    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_plot["timestamp"], y=df_plot["value"],
                        mode="lines", name="Series"
                    ))
                    for a in anomalies:
                        idx = int(a.get("timestamp_index", 0))
                        if 0 <= idx < len(df_plot):
                            fig.add_trace(go.Scatter(
                                x=[df_plot.iloc[idx]["timestamp"]],
                                y=[df_plot.iloc[idx]["value"]],
                                mode="markers",
                                marker=dict(color="red", size=12, symbol="x"),
                                name=f"Anomaly @ t={idx}",
                            ))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No anomalies detected.")

        with col_t:
            st.subheader("Trends")
            trends = result.get("trends", [])
            if trends:
                st.dataframe(pd.DataFrame(trends), hide_index=True, use_container_width=True)
            else:
                st.info("No distinct trends.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Alert Feed
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Alert Feed":
    st.header("Alert Feed")
    st.caption("Anomaly alerts generated from past analyses (medium + high severity).")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_domain = st.selectbox(
            "Domain", ["(all)", "default", "clinical", "financial", "iot"]
        )
    with col_f2:
        filter_sev = st.selectbox("Min severity", ["low", "medium", "high"], index=0)
    with col_f3:
        page_size = st.selectbox("Page size", [10, 20, 50], index=1)

    page_num = st.number_input("Page", min_value=1, value=1, step=1)
    offset = (page_num - 1) * page_size

    params: dict[str, Any] = {
        "severity_min": filter_sev,
        "limit": page_size,
        "offset": offset,
    }
    if filter_domain != "(all)":
        params["domain"] = filter_domain

    alerts_data = _api_get("/alerts", **params)

    if alerts_data is None:
        st.warning("Could not load alerts — is the API running?")
        st.stop()

    items = alerts_data.get("items", [])
    total = alerts_data.get("total", 0)

    st.markdown(f"**{total} total alerts** — showing {offset + 1}–{min(offset + page_size, total)}")

    if not items:
        st.info("No alerts match your filters.")
    else:
        for alert in items:
            with st.expander(
                f"{_severity_badge(alert['severity'])}  |  "
                f"`{alert['domain']}`  ·  {alert['anomaly_type']}  ·  "
                f"{alert['created_at'][:19]}"
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{alert['confidence']:.0%}")
                c2.metric("Timestamp index", alert["timestamp_index"])
                c3.metric("Acknowledged", "Yes" if alert["acknowledged"] else "No")

                st.markdown(f"**Description:** {alert['description']}")
                st.caption(f"Analysis ID: `{alert['analysis_id']}`")

                if st.button("View full analysis", key=f"view_{alert['id']}"):
                    analysis = _api_get(f"/analyses/{alert['analysis_id']}")
                    if analysis:
                        st.json(analysis, expanded=False)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Benchmark Results
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Benchmark Results":
    st.header("Benchmark: Visual vs Text Analysis")
    st.caption("Reproducing arXiv:2410.02637 — GPT-4o visual reasoning vs text serialisation.")

    results_path = Path("benchmarks/results.json")
    summary_path = Path("benchmarks/summary.json")

    if not results_path.exists():
        st.warning(
            "No benchmark results found. Run:\n"
            "```\npython benchmarks/compare_text_vs_visual.py --dry-run\n```"
        )
        st.stop()

    with results_path.open() as f:
        results: list[dict] = json.load(f)

    df_raw = pd.DataFrame(results)

    # Flatten nested text/visual dicts produced by the benchmark runner
    def _extract(col: str, key: str, df: pd.DataFrame) -> "pd.Series":
        if col in df.columns:
            return df[col].apply(lambda x: x.get(key, 0) if isinstance(x, dict) else 0)
        return pd.Series(0, index=df.index)

    df_raw["visual_confidence"] = _extract("visual", "confidence", df_raw)
    df_raw["text_confidence"] = _extract("text", "confidence", df_raw)
    df_raw["visual_latency_ms"] = _extract("visual", "latency_ms", df_raw)
    df_raw["text_latency_ms"] = _extract("text", "latency_ms", df_raw)
    if "confidence_delta" in df_raw.columns and "confidence_lift" not in df_raw.columns:
        df_raw["confidence_lift"] = df_raw["confidence_delta"]

    # ── Summary stats ──────────────────────────────────────────────────────────
    if summary_path.exists():
        with summary_path.open() as f:
            summary: dict = json.load(f)

        m1, m2, m3 = st.columns(3)
        overall = summary.get("overall", {})
        m1.metric(
            "Visual avg confidence",
            f"{overall.get('visual_mean_confidence', 0):.3f}",
            delta=f"+{overall.get('confidence_lift', 0):.3f} vs text",
        )
        m2.metric(
            "Text avg confidence",
            f"{overall.get('text_mean_confidence', 0):.3f}",
        )
        m3.metric("Windows benchmarked", overall.get("n_windows", len(df_raw)))

        st.divider()

    # ── Per-domain bar chart ───────────────────────────────────────────────────
    if "domain" in df_raw.columns:
        domain_grp = (
            df_raw.groupby("domain")[["visual_confidence", "text_confidence"]]
            .mean()
            .reset_index()
        )
        fig = go.Figure()
        fig.add_bar(
            x=domain_grp["domain"],
            y=domain_grp["visual_confidence"],
            name="Visual (GPT-4o)",
            marker_color="#4F8EF7",
        )
        fig.add_bar(
            x=domain_grp["domain"],
            y=domain_grp["text_confidence"],
            name="Text serialised",
            marker_color="#F78E4F",
        )
        fig.update_layout(
            barmode="group",
            title="Mean confidence by domain",
            xaxis_title="Domain",
            yaxis_title="Confidence (0-1)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Raw comparison table ───────────────────────────────────────────────────
    st.subheader("Raw results")
    display_cols = [
        c for c in ["domain", "window_id", "visual_confidence", "text_confidence",
                    "confidence_lift", "visual_latency_ms", "text_latency_ms"]
        if c in df_raw.columns
    ]
    st.dataframe(
        df_raw[display_cols].style.background_gradient(
            subset=[c for c in ["visual_confidence", "text_confidence"] if c in display_cols],
            cmap="RdYlGn",
        ),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MIMIC Demo
# ══════════════════════════════════════════════════════════════════════════════

elif page == "MIMIC Demo":
    st.header("MIMIC-III ICU Demo")
    st.caption(
        "Three anonymised ICU patient windows from MIMIC-III. "
        "Runs the full agentic pipeline and shows the step-by-step trace."
    )

    # ── Hardcoded anonymised MIMIC-like samples ────────────────────────────────
    _BASE = datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc)

    def _mimic_hr(seed: int, n: int = 60) -> list[dict]:
        """Simulated heart-rate window (bpm) with a transient tachycardia."""
        rng = np.random.default_rng(seed)
        t = np.arange(n)
        hr = 72 + rng.normal(0, 3, n)
        hr[30:38] += 42  # tachycardia episode
        return [
            {"timestamp": (_BASE + timedelta(minutes=i)).isoformat(), "value": float(v)}
            for i, v in enumerate(hr)
        ]

    def _mimic_spo2(seed: int, n: int = 60) -> list[dict]:
        """SpO2 (%) window with a desaturation dip."""
        rng = np.random.default_rng(seed)
        spo2 = 98 + rng.normal(0, 0.4, n)
        spo2[20:28] -= 8  # desaturation
        spo2 = np.clip(spo2, 80, 100)
        return [
            {"timestamp": (_BASE + timedelta(minutes=i)).isoformat(), "value": float(v)}
            for i, v in enumerate(spo2)
        ]

    def _mimic_bp(seed: int, n: int = 60) -> list[dict]:
        """Systolic BP (mmHg) window with a hypotensive event."""
        rng = np.random.default_rng(seed)
        bp = 120 + rng.normal(0, 5, n)  # noqa: assigned without seed reference intentionally
        bp[45:55] -= 35  # hypotension
        return [
            {"timestamp": (_BASE + timedelta(minutes=i)).isoformat(), "value": float(v)}
            for i, v in enumerate(bp)
        ]

    PATIENTS: list[dict] = [
        {
            "id": "ICU-A (anonymised)",
            "signal": "Heart Rate (bpm)",
            "domain": "clinical",
            "task": "anomaly_detection",
            "question": "Is there a tachycardia episode? When does it start and end?",
            "series": _mimic_hr(7),
        },
        {
            "id": "ICU-B (anonymised)",
            "signal": "SpO₂ (%)",
            "domain": "clinical",
            "task": "anomaly_detection",
            "question": "Identify any oxygen desaturation. Assess severity.",
            "series": _mimic_spo2(13),
        },
        {
            "id": "ICU-C (anonymised)",
            "signal": "Systolic BP (mmHg)",
            "domain": "clinical",
            "task": "anomaly_detection",
            "question": "Flag any hypotensive episodes (< 90 mmHg).",
            "series": _mimic_bp(21),
        },
    ]

    selected_label = st.selectbox(
        "Select patient window",
        [p["id"] for p in PATIENTS],
    )
    patient = next(p for p in PATIENTS if p["id"] == selected_label)

    # Preview the signal
    df_preview = pd.DataFrame(patient["series"])
    df_preview["timestamp"] = pd.to_datetime(df_preview["timestamp"])
    fig_prev = px.line(
        df_preview, x="timestamp", y="value",
        title=f"{patient['id']} — {patient['signal']}",
        labels={"value": patient["signal"], "timestamp": "Time"},
    )
    st.plotly_chart(fig_prev, use_container_width=True)

    st.info(f"**Clinical question:** {patient['question']}")

    run_col, _ = st.columns([1, 3])
    if run_col.button("Run Agentic Pipeline", type="primary"):
        payload = {
            "series": patient["series"],
            "domain": patient["domain"],
            "task": patient["task"],
            "question": patient["question"],
            "chain_of_thought": True,
            "plot_style": "line",
        }

        progress = st.progress(0, text="Initialising agent…")
        with st.spinner("Running LangGraph agentic loop…"):
            progress.progress(20, "Perceiving — rendering plot artifact…")
            data = _api_post("/analyze", payload, timeout=180)
            progress.progress(100, "Complete.")

        if not data:
            st.stop()

        result = data.get("result", {})

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence", f"{result.get('confidence', 0):.0%}")
        m2.metric("Latency", f"{data.get('processing_ms', 0)} ms")
        m3.metric("Cost", f"${data.get('cost_usd', 0):.4f}")
        m4.metric("Anomalies", len(result.get("anomalies", [])))

        st.subheader("Agent Reasoning Summary")
        st.write(result.get("summary", "—"))

        # Step-by-step agent trace
        steps = result.get("reasoning_steps", [])
        if steps:
            st.subheader("Step-by-Step Agent Trace")
            for i, step in enumerate(steps):
                icon = "🔍" if i == 0 else ("🧠" if i < len(steps) - 1 else "✅")
                with st.expander(
                    f"{icon} Iteration {step['step']}: {step['observation'][:70]}",
                    expanded=(i == 0),
                ):
                    st.markdown(f"**Observation:** {step['observation']}")
                    st.markdown(f"**Inference:** {step['inference']}")

        # Final analysis block
        final = data.get("final_analysis")
        if final:
            st.subheader("Final Agentic Output")
            fa_cols = st.columns(3)
            fa_cols[0].metric("Iterations taken", final.get("iterations_taken", "—"))
            fa_cols[1].metric("Escalated", "Yes" if final.get("escalated") else "No")
            tools = final.get("tools_used", [])
            fa_cols[2].metric("Tools used", len(tools))
            if tools:
                st.caption("Tools called: " + ", ".join(f"`{t}`" for t in tools))

        # Anomaly table
        anomalies = result.get("anomalies", [])
        if anomalies:
            st.subheader("Detected Anomalies")
            st.dataframe(
                pd.DataFrame(anomalies), hide_index=True, use_container_width=True
            )

        with st.expander("Raw API response"):
            st.json(data, expanded=False)
