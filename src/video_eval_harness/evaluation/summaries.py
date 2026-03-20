"""Summary generation and export for benchmark runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from ..log import get_logger
from ..schemas import SegmentLabelResult
from .metrics import compute_agreement_matrix, compute_model_summary, compute_sweep_metrics

logger = get_logger(__name__)
console = Console()


def print_run_summary(results: list[SegmentLabelResult], run_id: str) -> None:
    """Print a rich summary table for a benchmark run."""
    models = sorted({r.model_name for r in results})

    if not models:
        console.print("[yellow]No results to summarize.[/yellow]")
        return

    # Per-model summary
    summaries = [compute_model_summary(results, m) for m in models]

    table = Table(title=f"Run Summary: {run_id}", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Segments", justify="right")
    table.add_column("Parse OK", justify="right")
    table.add_column("Parse %", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("Avg Conf", justify="right")
    table.add_column("Est. Cost", justify="right")

    for s in summaries:
        table.add_row(
            s.model_name,
            str(s.total_segments),
            str(s.successful_parses),
            f"{s.parse_success_rate:.0%}",
            f"{s.avg_latency_ms:.0f}ms" if s.avg_latency_ms else "-",
            f"{s.p95_latency_ms:.0f}ms" if s.p95_latency_ms else "-",
            f"{s.avg_confidence:.2f}" if s.avg_confidence else "-",
            f"${s.total_estimated_cost:.4f}" if s.total_estimated_cost else "-",
        )

    console.print(table)

    # Agreement matrix
    agreement = compute_agreement_matrix(results)
    if len(models) > 1:
        ag_table = Table(title="Primary Action Agreement", show_lines=True)
        ag_table.add_column("", style="cyan")
        for m in models:
            ag_table.add_column(m[:20], justify="right")

        for m1 in models:
            row = [m1[:20]]
            for m2 in models:
                val = agreement.get(m1, {}).get(m2, 0)
                row.append(f"{val:.0%}")
            ag_table.add_row(*row)

        console.print(ag_table)

    # Sample outputs
    _print_sample_outputs(results, models)


def _print_sample_outputs(results: list[SegmentLabelResult], models: list[str], n: int = 3) -> None:
    """Print sample outputs side by side for quick inspection."""
    segments = sorted({r.segment_id for r in results})[:n]

    for seg_id in segments:
        seg_results = [r for r in results if r.segment_id == seg_id]
        if not seg_results:
            continue

        table = Table(title=f"Segment: {seg_id}", show_lines=True)
        table.add_column("Field", style="cyan")
        for m in models:
            table.add_column(m[:20])

        fields = ["primary_action", "description", "confidence"]
        for field in fields:
            row = [field]
            for m in models:
                mr = next((r for r in seg_results if r.model_name == m), None)
                if mr is None:
                    row.append("-")
                else:
                    val = getattr(mr, field, "-")
                    if val is None:
                        val = "-"
                    row.append(str(val)[:60])
            table.add_row(*row)

        console.print(table)


def print_sweep_summary(results: list[SegmentLabelResult], run_id: str) -> None:
    """Print a sweep-grouped Rich summary: model x variant matrix with metrics."""
    sweep_data = compute_sweep_metrics(results)
    cells = sweep_data["cells"]
    stability = sweep_data["stability"]
    agreement_by_variant = sweep_data["agreement_by_variant"]

    if not cells:
        console.print("[yellow]No sweep results to summarize.[/yellow]")
        return

    # Collect unique models and variants (preserving order)
    models = sorted({c.model_name for c in cells})
    variants = sorted({c.variant_label for c in cells})

    # Cell lookup
    cell_map = {(c.model_name, c.variant_label): c for c in cells}

    # Parse success rate matrix
    ptable = Table(title=f"Parse Success Rate — Run {run_id}", show_lines=True)
    ptable.add_column("Model", style="cyan", no_wrap=True)
    for v in variants:
        ptable.add_column(v, justify="right")

    for model in models:
        row = [model]
        for v in variants:
            c = cell_map.get((model, v))
            if c:
                rate = f"{c.parse_success_rate:.0%}"
                row.append(f"[green]{rate}[/green]" if c.parse_success_rate >= 0.9 else rate)
            else:
                row.append("-")
        ptable.add_row(*row)
    console.print(ptable)

    # Latency matrix
    ltable = Table(title="Avg Latency (ms)", show_lines=True)
    ltable.add_column("Model", style="cyan", no_wrap=True)
    for v in variants:
        ltable.add_column(v, justify="right")

    for model in models:
        row = [model]
        for v in variants:
            c = cell_map.get((model, v))
            row.append(f"{c.avg_latency_ms:.0f}" if c and c.avg_latency_ms else "-")
        ltable.add_row(*row)
    console.print(ltable)

    # Confidence matrix
    ctable = Table(title="Avg Confidence", show_lines=True)
    ctable.add_column("Model", style="cyan", no_wrap=True)
    for v in variants:
        ctable.add_column(v, justify="right")

    for model in models:
        row = [model]
        for v in variants:
            c = cell_map.get((model, v))
            row.append(f"{c.avg_confidence:.2f}" if c and c.avg_confidence else "-")
        ctable.add_row(*row)
    console.print(ctable)

    # Stability scores
    stable = Table(title="Model Stability Across Variants", show_lines=True)
    stable.add_column("Model", style="cyan", no_wrap=True)
    stable.add_column("Self-Agreement", justify="right")
    stable.add_column("Rank Stability", justify="right")
    stable.add_column("Rank Positions", justify="left")

    for s in stability:
        stable.add_row(
            s.model_name,
            f"{s.self_agreement:.0%}",
            f"{s.rank_stability:.2f}",
            ", ".join(f"#{r}" for r in s.rank_positions),
        )
    console.print(stable)

    # Per-variant agreement matrices
    for vlabel, agreement in agreement_by_variant.items():
        ag_models = sorted(agreement.keys())
        if len(ag_models) < 2:
            continue
        ag_table = Table(title=f"Agreement: {vlabel}", show_lines=True)
        ag_table.add_column("", style="cyan")
        for m in ag_models:
            ag_table.add_column(m[:20], justify="right")
        for m1 in ag_models:
            row = [m1[:20]]
            for m2 in ag_models:
                val = agreement.get(m1, {}).get(m2, 0)
                row.append(f"{val:.0%}")
            ag_table.add_row(*row)
        console.print(ag_table)


def results_to_dataframe(results: list[SegmentLabelResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame.

    Sweep columns (extraction_label, num_frames_used, sampling_method_used)
    are always included — they're empty strings / 0 for non-sweep runs.
    """
    records = []
    for r in results:
        d = r.model_dump()
        # Flatten lists to strings for tabular format
        d["secondary_actions"] = "; ".join(d.get("secondary_actions", []))
        d["objects"] = "; ".join(d.get("objects", []))
        d["uncertainty_flags"] = "; ".join(d.get("uncertainty_flags", []))
        d.pop("raw_response_text", None)  # Too large for table view
        records.append(d)
    return pd.DataFrame(records)


def export_results(
    results: list[SegmentLabelResult],
    output_dir: str | Path,
    run_id: str,
    formats: list[str] | None = None,
) -> list[Path]:
    """Export results to CSV and/or Parquet.

    Returns list of exported file paths.
    """
    if formats is None:
        formats = ["csv", "parquet"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    exported: list[Path] = []

    if "csv" in formats:
        csv_path = output_dir / f"{run_id}_results.csv"
        df.to_csv(csv_path, index=False)
        exported.append(csv_path)
        logger.info(f"Exported CSV: {csv_path}")

    if "parquet" in formats:
        parquet_path = output_dir / f"{run_id}_results.parquet"
        df.to_parquet(parquet_path, index=False)
        exported.append(parquet_path)
        logger.info(f"Exported Parquet: {parquet_path}")

    return exported
