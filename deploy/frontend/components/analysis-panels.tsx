import type { CSSProperties, ReactNode } from "react";

import type { ModelDeltaRow, ParseSuccessMatrix, SegmentComparisonSample } from "../lib/analysis";
import {
  formatDateTime,
  formatLatency,
  formatPercent,
  formatSignedLatency,
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
  className = "raw-section",
}: {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={className}>
      <h3>{title}</h3>
      {description ? <p className="chart-desc">{description}</p> : null}
      {children}
    </section>
  );
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
      className="agreement-section"
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
    <SectionCard title={title} description={description} className="agreement-section">
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
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>{leftLabel} Parse</th>
              <th>{rightLabel} Parse</th>
              <th>Delta</th>
              <th>{leftLabel} Latency</th>
              <th>{rightLabel} Latency</th>
              <th>Delta</th>
              <th>{leftLabel} Agreement</th>
              <th>{rightLabel} Agreement</th>
              <th>Delta</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.model_name}>
                <td>{row.model_name}</td>
                <td>{formatPercent(row.left_parse_rate)}</td>
                <td>{formatPercent(row.right_parse_rate)}</td>
                <td className={metricDeltaClass(row.parse_rate_delta)}>
                  {formatSignedPercentDelta(row.parse_rate_delta)}
                </td>
                <td>{formatLatency(row.left_latency_ms)}</td>
                <td>{formatLatency(row.right_latency_ms)}</td>
                <td className={metricDeltaClass(row.latency_delta_ms, true)}>
                  {formatSignedLatency(row.latency_delta_ms)}
                </td>
                <td>{formatPercent(row.left_agreement)}</td>
                <td>{formatPercent(row.right_agreement)}</td>
                <td className={metricDeltaClass(row.agreement_delta)}>
                  {formatSignedPercentDelta(row.agreement_delta)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </SectionCard>
  );
}

export function SegmentComparisonSamplesCard({
  title,
  description,
  samples,
}: {
  title: string;
  description?: string;
  samples: SegmentComparisonSample[];
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
        {samples.map((sample) => (
          <article key={sample.segment.segment_id} className="sample-card">
            <div className="sample-card-head">
              <div>
                <h4>{sample.segment.segment_id}</h4>
                <p className="sample-meta">
                  {formatTime(sample.segment.start_time_s)} - {formatTime(sample.segment.end_time_s)}
                  {sample.variant_label ? ` | ${sample.variant_label}` : ""}
                </p>
              </div>
              <span className="sample-agreement">
                Agreement {formatPercent(sample.mean_agreement)}
              </span>
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
                      <td>{result.primary_action || result.parse_error || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        ))}
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
}: {
  title: string;
  runId: string;
  createdAt: string;
  videoLabel: string;
  promptVersion: string;
  models: string[];
  segments: number;
  extraRows?: Array<{ label: string; value: string }>;
}) {
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
