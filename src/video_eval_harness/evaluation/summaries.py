"""Summary generation and export for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

from ..log import get_logger
from ..schemas import ModelRunSummary, SegmentLabelResult
from .metrics import compute_agreement_matrix, compute_model_summary

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


def results_to_dataframe(results: list[SegmentLabelResult]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame."""
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
