import type { CSSProperties, ReactNode } from "react";

import type {
  CoreComparisonRow,
  ModelDeltaRow,
  ParseSuccessMatrix,
  SegmentComparisonSample,
} from "../lib/analysis";
import {
  displayRunName,
  formatDateTime,
  formatLatency,
  formatPercent,
  formatMoney,
  formatSignedPercentDelta,
  formatTime,
} from "../lib/analysis";
import type { ModelStabilityScore } from "../lib/types";

function heatColor(value: number): string {
  if (value >= 0.8) {
    return "rgba(34, 197, 94, 0.25)";
  }
  if (value >= 0.5) {
    return "rgba(245, 158, 11, 0.2)";
  }
  if (value >= 0.3) {
    return "rgba(245, 158, 11, 0.12)";
  }
  return "rgba(239, 68, 68, 0.1)";
}

function deltaHeatColor(value: number): string {
  if (value >= 0.2) {
    return "rgba(34, 197, 94, 0.25)";
  }
  if (value > 0) {
    return "rgba(34, 197, 94, 0.12)";
  }
  if (value <= -0.2) {
    return "rgba(239, 68, 68, 0.18)";
  }
  if (value < 0) {
    return "rgba(239, 68, 68, 0.1)";
  }
  return "rgba(20, 40, 29, 0.05)";
}

function metricDeltaClass(value: number | null, lowerIsBetter = false): string {
  if (value == null || value === 0) {
    return "delta-neutral";
  }
  const isPositive = lowerIsBetter ? value < 0 : value > 0;
  return isPositive ? "delta-positive" : "delta-negative";
}

function cellStyle(value: number | null, deltaMode = false): CSSProperties | undefined {
  if (value == null) {
    return undefined;
  }
  return {
    background: deltaMode ? deltaHeatColor(value) : heatColor(value),
  };
}

function SectionCard({
  title,
  description,
  children,
  className = "visual-card section-card",
}: {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={className}>
      <div className="section-heading">
        <h3>{title}</h3>
        {description ? <p className="chart-desc">{description}</p> : null}
      </div>
      {children}
    </section>
  );
}

function primaryActionsDisagree(
  actions: Array<string | null | undefined>
): boolean {
  const normalized = [...new Set(actions.map((action) => action?.trim()).filter(Boolean))];
  return normalized.length > 1;
}

function deltaArrow(value: number | null, lowerIsBetter = false): string {
  if (value == null || value === 0) {
    return "\u2192";
  }
  const improved = lowerIsBetter ? value < 0 : value > 0;
  return improved ? "\u25b2" : "\u25bc";
}

function deltaValueLabel(value: number | null, kind: "percent" | "latency" = "percent"): string {
  if (value == null || value === 0) {
    return "No change";
  }
  if (kind === "latency") {
    return `${Math.abs(Math.round(value))} ms`;
  }
  return `${Math.abs(value * 100).toFixed(1)}%`;
}

function DeltaMetricCell({
  leftValue,
  rightValue,
  delta,
  lowerIsBetter = false,
  kind = "percent",
}: {
  leftValue: number | null;
  rightValue: number | null;
  delta: number | null;
  lowerIsBetter?: boolean;
  kind?: "percent" | "latency";
}) {
  const formatter = kind === "latency" ? formatLatency : formatPercent;
  return (
    <td className="delta-metric-cell">
      <div className="delta-metric-top">
        <strong>{formatter(rightValue)}</strong>
        <span className={metricDeltaClass(delta, lowerIsBetter)}>
          {deltaArrow(delta, lowerIsBetter)} {deltaValueLabel(delta, kind)}
        </span>
      </div>
      <p className="delta-metric-baseline">From {formatter(leftValue)}</p>
    </td>
  );
}

function bestOf(
  rows: CoreComparisonRow[],
  key: keyof Pick<
    CoreComparisonRow,
    "parse_rate" | "agreement" | "confidence" | "avg_latency_ms" | "total_cost" | "stability"
  >,
  preferLower = false
): number | null {
  const values = rows
    .map((row) => row[key])
    .filter((value): value is number => value != null);
  if (values.length === 0) {
    return null;
  }
  return preferLower ? Math.min(...values) : Math.max(...values);
}

function worstOf(
  rows: CoreComparisonRow[],
  key: keyof Pick<
    CoreComparisonRow,
    "parse_rate" | "agreement" | "confidence" | "avg_latency_ms" | "total_cost" | "stability"
  >,
  preferLower = false
): number | null {
  const values = rows
    .map((row) => row[key])
    .filter((value): value is number => value != null);
  if (values.length === 0) {
    return null;
  }
  return preferLower ? Math.max(...values) : Math.min(...values);
}

function scoreClass(
  value: number | null,
  best: number | null,
  worst: number | null
): string | undefined {
  if (value == null) {
    return undefined;
  }
  if (best != null && value === best) {
    return "score-best";
  }
  if (worst != null && value === worst) {
    return "score-worst";
  }
  return undefined;
}

export function AgreementMatrixCard({
  title,
  description,
  matrix,
}: {
  title: string;
  description?: string;
  matrix: Record<string, Record<string, number>>;
}) {
  const models = Object.keys(matrix).sort();
  if (models.length === 0) {
    return null;
  }

  return (
    <SectionCard
      title={title}
      description={description ?? "Pairwise primary-action agreement across model outputs."}
      className="visual-card agreement-section"
    >
      <div className="matrix-scroll">
        <table className="agreement-table">
          <thead>
            <tr>
              <th />
              {models.map((model) => (
                <th key={model}>{model}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((rowModel) => (
              <tr key={rowModel}>
                <td className="matrix-row-label">{rowModel}</td>
                {models.map((columnModel) => {
                  const value = matrix[rowModel]?.[columnModel] ?? 0;
                  return (
                    <td
                      key={`${rowModel}-${columnModel}`}
                      className="matrix-cell mono"
                      style={cellStyle(value)}
                    >
                      {formatPercent(value)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}

export function CoreComparisonTable({
  rows,
}: {
  rows: CoreComparisonRow[];
}) {
  if (rows.length === 0) {
    return null;
  }

  const bestParse = bestOf(rows, "parse_rate");
  const worstParse = worstOf(rows, "parse_rate");
  const bestAgreement = bestOf(rows, "agreement");
  const worstAgreement = worstOf(rows, "agreement");
  const bestConfidence = bestOf(rows, "confidence");
  const worstConfidence = worstOf(rows, "confidence");
  const bestLatency = bestOf(rows, "avg_latency_ms", true);
  const worstLatency = worstOf(rows, "avg_latency_ms", true);
  const bestCost = bestOf(rows, "total_cost", true);
  const worstCost = worstOf(rows, "total_cost", true);
  const bestStability = bestOf(rows, "stability");
  const worstStability = worstOf(rows, "stability");

  return (
    <section className="raw-section">
      <h3>Core Comparison</h3>
      <p className="chart-desc">
        Sorted by agreement so the strongest consensus model surfaces first.
      </p>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Parse %</th>
              <th>Agreement</th>
              <th>Confidence</th>
              <th>Avg Latency</th>
              <th>Cost</th>
              <th>Stability</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model_name}>
                <td>{row.model_name}</td>
                <td className={scoreClass(row.parse_rate, bestParse, worstParse)}>
                  {formatPercent(row.parse_rate)}
                </td>
                <td className={scoreClass(row.agreement, bestAgreement, worstAgreement)}>
                  {formatPercent(row.agreement)}
                </td>
                <td className={scoreClass(row.confidence, bestConfidence, worstConfidence)}>
                  {row.confidence == null ? "-" : row.confidence.toFixed(3)}
                </td>
                <td className={scoreClass(row.avg_latency_ms, bestLatency, worstLatency)}>
                  {formatLatency(row.avg_latency_ms)}
                </td>
                <td className={scoreClass(row.total_cost, bestCost, worstCost)}>
                  {formatMoney(row.total_cost)}
                </td>
                <td className={scoreClass(row.stability, bestStability, worstStability)}>
                  {row.stability == null ? "-" : row.stability.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function VariantHeatmapCard({
  title,
  description,
  models,
  variants,
  matrix,
  deltaMode = false,
}: {
  title: string;
  description?: string;
  models: string[];
  variants: string[];
  matrix: ParseSuccessMatrix;
  deltaMode?: boolean;
}) {
  if (models.length === 0 || variants.length === 0) {
    return null;
  }

  return (
    <SectionCard title={title} description={description} className="visual-card agreement-section">
      <div className="matrix-scroll">
        <table className="agreement-table">
          <thead>
            <tr>
              <th />
              {variants.map((variant) => (
                <th key={variant}>{variant}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model}>
                <td className="matrix-row-label">{model}</td>
                {variants.map((variant) => {
                  const value = matrix[model]?.[variant] ?? null;
                  return (
                    <td
                      key={`${model}-${variant}`}
                      className="matrix-cell mono"
                      style={cellStyle(value, deltaMode)}
                    >
                      {deltaMode
                        ? formatSignedPercentDelta(value)
                        : formatPercent(value)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}

export function StabilityTableCard({
  title,
  description,
  stability,
}: {
  title: string;
  description?: string;
  stability: ModelStabilityScore[];
}) {
  if (stability.length === 0) {
    return null;
  }

  return (
    <SectionCard title={title} description={description}>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Self Agreement</th>
              <th>Rank Stability</th>
              <th>Rank Positions</th>
            </tr>
          </thead>
          <tbody>
            {stability.map((entry) => (
              <tr key={entry.model_name}>
                <td>{entry.model_name}</td>
                <td>{formatPercent(entry.self_agreement)}</td>
                <td>{entry.rank_stability.toFixed(3)}</td>
                <td>{entry.rank_positions.map((value) => `#${value}`).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}

export function MetricDeltaTableCard({
  title,
  description,
  rows,
  leftLabel,
  rightLabel,
}: {
  title: string;
  description?: string;
  rows: ModelDeltaRow[];
  leftLabel: string;
  rightLabel: string;
}) {
  if (rows.length === 0) {
    return null;
  }

  return (
    <SectionCard title={title} description={description}>
      <div className="table-scroll">
        <table className="data-table delta-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Parse Rate</th>
              <th>Agreement</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model_name}>
                <td>{row.model_name}</td>
                <DeltaMetricCell
                  leftValue={row.left_parse_rate}
                  rightValue={row.right_parse_rate}
                  delta={row.parse_rate_delta}
                />
                <DeltaMetricCell
                  leftValue={row.left_agreement}
                  rightValue={row.right_agreement}
                  delta={row.agreement_delta}
                />
                <DeltaMetricCell
                  leftValue={row.left_latency_ms}
                  rightValue={row.right_latency_ms}
                  delta={row.latency_delta_ms}
                  lowerIsBetter
                  kind="latency"
                />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="table-note">
        {leftLabel} is the baseline. {rightLabel} is the comparison run.
      </p>
    </SectionCard>
  );
}

export function SegmentComparisonSamplesCard({
  title,
  description,
  samples,
  formatSampleTitle,
  formatSampleMeta,
}: {
  title: string;
  description?: string;
  samples: SegmentComparisonSample[];
  formatSampleTitle?: (sample: SegmentComparisonSample) => string;
  formatSampleMeta?: (sample: SegmentComparisonSample) => string;
}) {
  if (samples.length === 0) {
    return (
      <SectionCard title={title} description={description}>
        <p className="empty-state">No segment comparisons available for this run.</p>
      </SectionCard>
    );
  }

  return (
    <SectionCard title={title} description={description}>
      <div className="sample-grid">
        {samples.map((sample) => {
          const disagree = primaryActionsDisagree(
            sample.results.map((result) => result.primary_action)
          );
          return (
            <article key={sample.segment.segment_id} className="sample-card">
              <div className="sample-card-head">
                <div>
                  <h4>{formatSampleTitle ? formatSampleTitle(sample) : sample.segment.segment_id}</h4>
                  <p className="sample-meta">
                    {formatSampleMeta
                      ? formatSampleMeta(sample)
                      : `${formatTime(sample.segment.start_time_s)} - ${formatTime(sample.segment.end_time_s)}${sample.variant_label ? ` \u00b7 ${sample.variant_label}` : ""}`}
                  </p>
                </div>
                <div className="sample-card-badges">
                  {disagree ? <span className="warning-pill">Models disagree</span> : null}
                  <span className="sample-agreement">
                    Agreement {formatPercent(sample.mean_agreement)}
                  </span>
                </div>
              </div>
              <div className="table-scroll">
                <table className="data-table compact-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Parsed</th>
                      <th>Primary Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sample.results.map((result) => (
                      <tr key={`${sample.segment.segment_id}-${result.model_name}`}>
                        <td>{result.model_name}</td>
                        <td>
                          <span className={`parse-badge small ${result.parsed_success ? "ok" : "fail"}`}>
                            {result.parsed_success ? "yes" : "no"}
                          </span>
                        </td>
                        <td className="sample-primary-action">
                          {result.primary_action || result.parse_error || "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          );
        })}
      </div>
    </SectionCard>
  );
}

export function RunMetadataCard({
  title,
  runId,
  createdAt,
  videoLabel,
  promptVersion,
  models,
  segments,
  extraRows = [],
  compact = false,
  compactText,
}: {
  title: string;
  runId: string;
  createdAt: string;
  videoLabel: string;
  promptVersion: string;
  models: string[];
  segments: number;
  extraRows?: Array<{ label: string; value: string }>;
  compact?: boolean;
  compactText?: string;
}) {
  if (compact) {
    return (
      <p className="page-breadcrumb">
        {compactText ??
          [
            displayRunName(runId, createdAt),
            `${models.length} ${models.length === 1 ? "model" : "models"}`,
            videoLabel,
            `${segments} ${segments === 1 ? "segment" : "segments"}`,
          ].join(" \u00b7 ")}
      </p>
    );
  }

  return (
    <SectionCard title={title}>
      <div className="table-scroll">
        <table className="data-table">
          <tbody>
            <tr>
              <th>Run ID</th>
              <td className="mono">{runId}</td>
            </tr>
            <tr>
              <th>Date</th>
              <td>{formatDateTime(createdAt)}</td>
            </tr>
            <tr>
              <th>Video</th>
              <td>{videoLabel}</td>
            </tr>
            <tr>
              <th>Prompt Template</th>
              <td>{promptVersion}</td>
            </tr>
            <tr>
              <th>Models</th>
              <td>{models.join(", ")}</td>
            </tr>
            <tr>
              <th>Segments</th>
              <td>{segments}</td>
            </tr>
            {extraRows.map((row) => (
              <tr key={row.label}>
                <th>{row.label}</th>
                <td>{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}
