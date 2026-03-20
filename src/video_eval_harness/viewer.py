"""Minimal Streamlit viewer for benchmark results.

Run with: streamlit run src/video_eval_harness/viewer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install video-eval-harness[ui]")
    sys.exit(1)

import pandas as pd

# Add src to path if running directly
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from video_eval_harness.storage import Storage
from video_eval_harness.evaluation.summaries import results_to_dataframe
from video_eval_harness.evaluation.metrics import compute_model_summary, compute_agreement_matrix


def main() -> None:
    st.set_page_config(page_title="VBench - Video Eval Harness", layout="wide")
    st.title("VBench - Video Evaluation Harness")

    # Sidebar: artifacts directory
    artifacts_dir = st.sidebar.text_input("Artifacts Directory", value="artifacts")

    try:
        storage = Storage(artifacts_dir)
    except Exception as e:
        st.error(f"Cannot open storage at '{artifacts_dir}': {e}")
        return

    # List runs
    runs = storage.list_runs()
    if not runs:
        st.warning("No benchmark runs found. Run `vbench run-benchmark` first.")
        return

    run_options = {r.run_id: f"{r.run_id} ({r.created_at[:16]}, {len(r.models)} models)" for r in runs}
    selected_run_id = st.sidebar.selectbox("Select Run", options=list(run_options.keys()), format_func=lambda x: run_options[x])

    if not selected_run_id:
        return

    run_config = storage.get_run(selected_run_id)
    results = storage.get_run_results(selected_run_id)

    if not results:
        st.warning("No results for this run.")
        return

    # Run info
    st.header("Run Configuration")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Run ID", selected_run_id[:16])
    col2.metric("Models", len(run_config.models))
    col3.metric("Videos", len(run_config.video_ids))
    col4.metric("Prompt", run_config.prompt_version)

    # Model summaries
    st.header("Model Performance Summary")
    models = sorted({r.model_name for r in results})
    summaries = [compute_model_summary(results, m) for m in models]

    summary_data = []
    for s in summaries:
        summary_data.append({
            "Model": s.model_name,
            "Segments": s.total_segments,
            "Parse OK": s.successful_parses,
            "Parse %": f"{s.parse_success_rate:.0%}",
            "Avg Latency (ms)": f"{s.avg_latency_ms:.0f}" if s.avg_latency_ms else "-",
            "P95 Latency (ms)": f"{s.p95_latency_ms:.0f}" if s.p95_latency_ms else "-",
            "Avg Confidence": f"{s.avg_confidence:.2f}" if s.avg_confidence else "-",
            "Est. Cost": f"${s.total_estimated_cost:.4f}" if s.total_estimated_cost else "-",
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    # Agreement matrix
    if len(models) > 1:
        st.header("Primary Action Agreement Matrix")
        agreement = compute_agreement_matrix(results)
        ag_data = {}
        for m1 in models:
            ag_data[m1] = {m2: f"{agreement.get(m1, {}).get(m2, 0):.0%}" for m2 in models}
        st.dataframe(pd.DataFrame(ag_data, index=models), use_container_width=True)

    # Latency comparison chart
    st.header("Latency Distribution")
    df = results_to_dataframe(results)
    if "latency_ms" in df.columns:
        latency_df = df[df["latency_ms"].notna() & (df["latency_ms"] > 0)]
        if not latency_df.empty:
            try:
                import plotly.express as px
                fig = px.box(latency_df, x="model_name", y="latency_ms", title="Latency by Model (ms)")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(latency_df.groupby("model_name")["latency_ms"].mean())

    # Per-segment comparison
    st.header("Segment-Level Comparison")
    segments = sorted({r.segment_id for r in results})
    selected_segment = st.selectbox("Select Segment", segments)

    if selected_segment:
        seg_results = [r for r in results if r.segment_id == selected_segment]

        cols = st.columns(len(models))
        for i, model in enumerate(models):
            mr = next((r for r in seg_results if r.model_name == model), None)
            with cols[i]:
                st.subheader(model)
                if mr is None:
                    st.write("No result")
                elif not mr.parsed_success:
                    st.error(f"Parse failed: {mr.parse_error}")
                else:
                    st.write(f"**Action:** {mr.primary_action or '-'}")
                    st.write(f"**Description:** {mr.description or '-'}")
                    st.write(f"**Confidence:** {mr.confidence or '-'}")
                    if mr.secondary_actions:
                        st.write(f"**Secondary:** {', '.join(mr.secondary_actions)}")
                    if mr.objects:
                        st.write(f"**Objects:** {', '.join(mr.objects)}")
                    if mr.environment_context:
                        st.write(f"**Environment:** {mr.environment_context}")

        # Show extracted frames if available
        frames = storage.get_extracted_frames(selected_segment)
        if frames and frames.frame_paths:
            st.subheader("Extracted Frames")
            frame_cols = st.columns(min(len(frames.frame_paths), 4))
            for j, fp in enumerate(frames.frame_paths):
                fp_path = Path(fp)
                if fp_path.exists():
                    frame_cols[j % len(frame_cols)].image(str(fp_path), caption=f"t={frames.frame_timestamps_s[j]:.1f}s")

    # Raw data export
    st.header("Raw Data")
    if st.checkbox("Show raw results table"):
        st.dataframe(df, use_container_width=True)

    if st.button("Download CSV"):
        csv = df.to_csv(index=False)
        st.download_button("Download", csv, file_name=f"{selected_run_id}_results.csv", mime="text/csv")


if __name__ == "__main__":
    main()
