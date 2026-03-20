"""Streamlit viewer for benchmark and sweep results.

Run with:
  streamlit run src/video_eval_harness/viewer.py
"""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install video-eval-harness[ui]")
    sys.exit(1)

import pandas as pd


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _load_backend() -> dict[str, Any]:
    _ensure_src_on_path()

    from video_eval_harness.evaluation.metrics import (
        compute_agreement_matrix,
        compute_model_summary,
        compute_sweep_metrics,
    )
    from video_eval_harness.evaluation.summaries import results_to_dataframe
    from video_eval_harness.schemas import ExtractedFrames, SegmentLabelResult
    from video_eval_harness.storage import Storage

    return {
        "Storage": Storage,
        "ExtractedFrames": ExtractedFrames,
        "SegmentLabelResult": SegmentLabelResult,
        "compute_agreement_matrix": compute_agreement_matrix,
        "compute_model_summary": compute_model_summary,
        "compute_sweep_metrics": compute_sweep_metrics,
        "results_to_dataframe": results_to_dataframe,
    }


def _variant_id(result) -> str:
    return result.extraction_variant_id or "default"


def _variant_label(result) -> str:
    if result.extraction_variant_id:
        return result.extraction_label or result.extraction_variant_id
    return "default"


@st.cache_data(show_spinner=False)
def _list_runs(artifacts_dir: str) -> list[dict[str, Any]]:
    backend = _load_backend()
    storage = backend["Storage"](artifacts_dir)
    return [run.model_dump() for run in storage.list_runs()]


@st.cache_data(show_spinner=False)
def _load_run_bundle(artifacts_dir: str, run_id: str) -> dict[str, Any]:
    backend = _load_backend()
    storage = backend["Storage"](artifacts_dir)

    run_config = storage.get_run(run_id)
    results = storage.get_run_results(run_id)
    return {
        "run_config": run_config.model_dump() if run_config is not None else None,
        "results": [result.model_dump() for result in results],
    }


@st.cache_data(show_spinner=False)
def _load_frames_payload(
    artifacts_dir: str, video_id: str, segment_id: str, variant_id: str | None
) -> dict[str, Any] | None:
    backend = _load_backend()
    storage = backend["Storage"](artifacts_dir)

    if variant_id and variant_id != "default":
        frames_dir = storage.frames_dir(video_id, segment_id, variant_id)
        manifest_path = frames_dir / "metadata.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))

    frames = storage.get_extracted_frames(segment_id)
    if frames is None:
        return None
    return frames.model_dump()


def _style_rate_matrix(df: pd.DataFrame):
    styler = df.style.format("{:.0%}")
    if importlib.util.find_spec("matplotlib") is not None:
        styler = styler.background_gradient(
            cmap="RdYlGn", axis=None, vmin=0.0, vmax=1.0
        )
    return styler


def main() -> None:
    backend = _load_backend()
    Storage = backend["Storage"]
    SegmentLabelResult = backend["SegmentLabelResult"]
    compute_model_summary = backend["compute_model_summary"]
    compute_agreement_matrix = backend["compute_agreement_matrix"]
    compute_sweep_metrics = backend["compute_sweep_metrics"]
    results_to_dataframe = backend["results_to_dataframe"]

    st.set_page_config(page_title="VBench - Video Eval Harness", layout="wide")
    st.title("VBench - Video Evaluation Harness")

    artifacts_dir = st.sidebar.text_input("Artifacts Directory", value="artifacts")
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()

    try:
        Storage(artifacts_dir)
    except Exception as exc:
        st.error(f"Cannot open storage at '{artifacts_dir}': {exc}")
        return

    runs = _list_runs(artifacts_dir)
    if not runs:
        st.warning("No benchmark runs found. Run `vbench run-benchmark` first.")
        return

    run_options = {
        run["run_id"]: (
            f"{run['run_id']} ({str(run['created_at'])[:16]}, "
            f"{len(run.get('models', []))} models)"
        )
        for run in runs
    }
    selected_run_id = st.sidebar.selectbox(
        "Select Run",
        options=list(run_options.keys()),
        format_func=lambda run_id: run_options[run_id],
    )
    if not selected_run_id:
        return

    bundle = _load_run_bundle(artifacts_dir, selected_run_id)
    run_config = bundle["run_config"]
    results = [SegmentLabelResult(**item) for item in bundle["results"]]

    if run_config is None:
        st.warning("Run configuration could not be loaded.")
        return
    if not results:
        st.warning("No results for this run.")
        return

    has_sweep = any((result.extraction_variant_id or "").strip() for result in results)
    variant_label_to_id = {}
    for result in results:
        variant_label_to_id.setdefault(_variant_label(result), _variant_id(result))
    variant_labels = sorted(variant_label_to_id.keys())

    st.header("Run Configuration")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Run ID", selected_run_id[:16])
    col2.metric("Models", len(run_config.get("models", [])))
    col3.metric("Videos", len(run_config.get("video_ids", [])))
    col4.metric("Prompt", run_config.get("prompt_version", "-"))

    st.header("Model Performance Summary")
    models = sorted({result.model_name for result in results})
    summaries = [compute_model_summary(results, model_name) for model_name in models]
    summary_df = pd.DataFrame(
        [
            {
                "Model": summary.model_name,
                "Segments": summary.total_segments,
                "Parse OK": summary.successful_parses,
                "Parse %": summary.parse_success_rate,
                "Avg Latency (ms)": summary.avg_latency_ms,
                "P95 Latency (ms)": summary.p95_latency_ms,
                "Avg Confidence": summary.avg_confidence,
                "Est. Cost": summary.total_estimated_cost,
            }
            for summary in summaries
        ]
    )
    st.dataframe(
        summary_df.style.format(
            {
                "Parse %": "{:.0%}",
                "Avg Latency (ms)": "{:.0f}",
                "P95 Latency (ms)": "{:.0f}",
                "Avg Confidence": "{:.2f}",
                "Est. Cost": "${:.4f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
    )

    if len(models) > 1:
        st.header("Primary Action Agreement Matrix")
        agreement = compute_agreement_matrix(results)
        agreement_df = pd.DataFrame(agreement).reindex(index=models, columns=models)
        st.dataframe(
            _style_rate_matrix(agreement_df),
            use_container_width=True,
        )

    if has_sweep:
        st.header("Sweep Analysis")
        sweep_data = compute_sweep_metrics(results)
        cells = sweep_data["cells"]
        stability_scores = sweep_data["stability"]
        agreement_by_variant = sweep_data["agreement_by_variant"]

        parse_success_matrix = pd.DataFrame(
            [
                {
                    "Model": cell.model_name,
                    "Variant": cell.variant_label,
                    "Parse Success": cell.parse_success_rate,
                }
                for cell in cells
            ]
        ).pivot(index="Model", columns="Variant", values="Parse Success")
        parse_success_matrix = parse_success_matrix.reindex(
            index=models, columns=sorted(parse_success_matrix.columns)
        )
        st.subheader("Model x Variant Parse Success")
        st.dataframe(_style_rate_matrix(parse_success_matrix), use_container_width=True)

        st.subheader("Stability Scores")
        stability_df = pd.DataFrame(
            [
                {
                    "Model": score.model_name,
                    "Self Agreement": score.self_agreement,
                    "Rank Stability": score.rank_stability,
                    "Rank Positions": ", ".join(str(rank) for rank in score.rank_positions),
                }
                for score in stability_scores
            ]
        )
        st.dataframe(
            stability_df.style.format(
                {"Self Agreement": "{:.0%}", "Rank Stability": "{:.2f}"}, na_rep="-"
            ),
            use_container_width=True,
        )

        st.subheader("Agreement by Variant")
        for variant_label in sorted(agreement_by_variant.keys()):
            with st.expander(f"Agreement Matrix - {variant_label}", expanded=False):
                matrix_df = pd.DataFrame(agreement_by_variant[variant_label]).reindex(
                    index=models, columns=models
                )
                st.dataframe(
                    _style_rate_matrix(matrix_df),
                    use_container_width=True,
                )

    st.header("Latency Distribution")
    df = results_to_dataframe(results)
    if "latency_ms" in df.columns:
        latency_df = df[df["latency_ms"].notna() & (df["latency_ms"] > 0)]
        if not latency_df.empty:
            try:
                import plotly.express as px

                fig = px.box(
                    latency_df,
                    x="model_name",
                    y="latency_ms",
                    color="extraction_label" if has_sweep and "extraction_label" in latency_df.columns else None,
                    title="Latency by Model (ms)",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(latency_df.groupby("model_name")["latency_ms"].mean())

    st.header("Segment-Level Comparison")
    segments = sorted({result.segment_id for result in results})
    selected_segment = st.selectbox("Select Segment", segments)

    selected_variant_label = "All variants"
    if has_sweep:
        selected_variant_label = st.selectbox(
            "Variant Filter",
            options=["All variants"] + variant_labels,
            help="Filter segment comparison cards to a single extraction variant.",
        )

    if selected_segment:
        segment_results = [result for result in results if result.segment_id == selected_segment]
        if has_sweep and selected_variant_label != "All variants":
            segment_results = [
                result
                for result in segment_results
                if _variant_label(result) == selected_variant_label
            ]

        display_results = sorted(
            segment_results,
            key=lambda result: (_variant_label(result), result.model_name),
        )
        column_count = max(1, min(len(display_results), 4))
        columns = st.columns(column_count)

        for index, result in enumerate(display_results):
            with columns[index % column_count]:
                title = result.model_name
                if has_sweep:
                    title = f"{result.model_name} [{_variant_label(result)}]"
                st.subheader(title)
                if not result.parsed_success:
                    st.error(f"Parse failed: {result.parse_error}")
                else:
                    st.write(f"**Action:** {result.primary_action or '-'}")
                    st.write(f"**Description:** {result.description or '-'}")
                    st.write(f"**Confidence:** {result.confidence or '-'}")
                    if result.secondary_actions:
                        st.write(f"**Secondary:** {', '.join(result.secondary_actions)}")
                    if result.objects:
                        st.write(f"**Objects:** {', '.join(result.objects)}")
                    if result.environment_context:
                        st.write(f"**Environment:** {result.environment_context}")

        frame_payload = None
        if display_results:
            selected_variant_id = None
            if has_sweep and selected_variant_label != "All variants":
                selected_variant_id = variant_label_to_id[selected_variant_label]
            frame_payload = _load_frames_payload(
                artifacts_dir,
                display_results[0].video_id,
                selected_segment,
                selected_variant_id,
            )

        if frame_payload and frame_payload.get("frame_paths"):
            st.subheader("Extracted Frames")
            frame_paths = frame_payload.get("frame_paths", [])
            frame_timestamps = frame_payload.get("frame_timestamps_s", [])
            frame_columns = st.columns(min(len(frame_paths), 4))
            for idx, frame_path in enumerate(frame_paths):
                path = Path(frame_path)
                if path.exists():
                    caption = None
                    if idx < len(frame_timestamps):
                        caption = f"t={frame_timestamps[idx]:.1f}s"
                    frame_columns[idx % len(frame_columns)].image(str(path), caption=caption)

    st.header("Raw Data")
    if st.checkbox("Show raw results table"):
        st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv,
        file_name=f"{selected_run_id}_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
