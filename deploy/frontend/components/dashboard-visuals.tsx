"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { CostBreakdown, CoreComparisonRow } from "../lib/analysis";
import {
  agreementPerDollarScore,
  bestValueModel,
  fastestModel,
  formatLatency,
  formatMoney,
  formatPercent,
  hasAccuracy,
  modelColor,
  overallModelScore,
} from "../lib/analysis";
import type { RunPayload, SweepMetrics } from "../lib/types";
import { InfoTooltip } from "./info-tooltip";

type DashboardVisualsProps = {
  run: RunPayload;
  rows: CoreComparisonRow[];
  costBreakdown: CostBreakdown | null;
  sweepData: SweepMetrics | null;
};

type SortKey =
  | "agreement"
  | "accuracy"
  | "confidence"
  | "latency"
  | "cost"
  | "stability"
  | "model";
type SortDirection = "asc" | "desc";

type ModelView = CoreComparisonRow & {
  color: string;
  agreement_pct: number | null;
  accuracy_pct: number | null;
  primary_score: number | null;
  primary_score_pct: number | null;
  cost_per_segment: number | null;
  value_score: number | null;
};

type CostDetail = {
  model_name: string;
  color: string;
  total_cost: number | null;
  cost_per_segment: number | null;
  variants: Array<{ label: string; total_cost: number }>;
};

function tooltipStyle() {
  return {
    background: "rgba(10, 10, 15, 0.96)",
    border: "1px solid rgba(148, 163, 184, 0.22)",
    borderRadius: 16,
    color: "#f8fafc",
    boxShadow: "0 18px 40px rgba(0, 0, 0, 0.32)",
  };
}

function heatmapColor(value: number): string {
  if (value >= 0.85) {
    return "rgba(34, 197, 94, 0.86)";
  }
  if (value >= 0.65) {
    return "rgba(163, 230, 53, 0.8)";
  }
  if (value >= 0.45) {
    return "rgba(245, 158, 11, 0.76)";
  }
  if (value >= 0.25) {
    return "rgba(249, 115, 22, 0.72)";
  }
  return "rgba(239, 68, 68, 0.76)";
}

function latencyColor(index: number, total: number): string {
  if (total <= 1) {
    return "#22c55e";
  }
  const percentile = index / (total - 1);
  if (percentile <= 0.34) {
    return "#22c55e";
  }
  if (percentile <= 0.67) {
    return "#f59e0b";
  }
  return "#ef4444";
}

function numericValue(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    return Number(value);
  }
  if (Array.isArray(value) && value.length > 0) {
    return numericValue(value[0]);
  }
  return 0;
}

function median(values: number[]): number | null {
  if (!values.length) {
    return null;
  }
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  if (ordered.length % 2 === 0) {
    return (ordered[middle - 1] + ordered[middle]) / 2;
  }
  return ordered[middle];
}

function domainWithPadding(values: number[], minimumPadding: number): [number, number] | null {
  if (!values.length) {
    return null;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Math.max(max - min, minimumPadding);
  const padding = Math.max(spread * 0.18, minimumPadding);
  return [Math.max(0, min - padding), max + padding];
}

function formatConfidence(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return value.toFixed(2);
}

function metricValue(
  row: Pick<ModelView, "accuracy" | "agreement">,
  accuracyAvailable: boolean
): number | null {
  return accuracyAvailable ? row.accuracy : row.agreement;
}

function metricLabel(accuracyAvailable: boolean): string {
  return accuracyAvailable ? "accuracy" : "agreement";
}

function defaultDirectionFor(key: SortKey): SortDirection {
  if (key === "latency" || key === "cost" || key === "model") {
    return "asc";
  }
  return "desc";
}

function compareRows(
  left: ModelView,
  right: ModelView,
  key: SortKey,
  direction: SortDirection
): number {
  const multiplier = direction === "asc" ? 1 : -1;
  if (key === "model") {
    return multiplier * left.model_name.localeCompare(right.model_name);
  }

  const leftValue =
    key === "latency"
      ? left.avg_latency_ms
      : key === "cost"
        ? left.cost_per_segment
        : key === "stability"
          ? left.stability
          : key === "confidence"
            ? left.confidence
            : key === "accuracy"
              ? left.accuracy
              : left.agreement;
  const rightValue =
    key === "latency"
      ? right.avg_latency_ms
      : key === "cost"
        ? right.cost_per_segment
        : key === "stability"
          ? right.stability
          : key === "confidence"
            ? right.confidence
            : key === "accuracy"
              ? right.accuracy
              : right.agreement;

  const normalizedLeft =
    leftValue == null
      ? direction === "asc"
        ? Number.POSITIVE_INFINITY
        : Number.NEGATIVE_INFINITY
      : leftValue;
  const normalizedRight =
    rightValue == null
      ? direction === "asc"
        ? Number.POSITIVE_INFINITY
        : Number.NEGATIVE_INFINITY
      : rightValue;

  if (normalizedLeft !== normalizedRight) {
    return multiplier * (normalizedLeft - normalizedRight);
  }
  return left.model_name.localeCompare(right.model_name);
}

function metricExtremes(rows: ModelView[], key: SortKey): { best: number | null; worst: number | null } {
  const values = rows
    .map((row) =>
      key === "agreement"
        ? row.agreement
        : key === "accuracy"
          ? row.accuracy
          : key === "confidence"
            ? row.confidence
            : key === "latency"
              ? row.avg_latency_ms
              : key === "cost"
                ? row.cost_per_segment
                : row.stability
    )
    .filter((value): value is number => value != null);
  if (!values.length) {
    return { best: null, worst: null };
  }
  return key === "latency" || key === "cost"
    ? { best: Math.min(...values), worst: Math.max(...values) }
    : { best: Math.max(...values), worst: Math.min(...values) };
}

function cellClass(value: number | null | undefined, best: number | null, worst: number | null): string {
  if (value == null || best == null || worst == null || best === worst) {
    return "";
  }
  if (Math.abs(value - best) < 1e-9) {
    return "score-best";
  }
  if (Math.abs(value - worst) < 1e-9) {
    return "score-worst";
  }
  return "";
}

function QuestionHeader({
  eyebrow,
  title,
  description,
  accentColor,
}: {
  eyebrow: string;
  title: string;
  description: string;
  accentColor: string;
}) {
  return (
    <div className="section-heading">
      <p className="section-eyebrow" style={{ color: accentColor }}>
        {eyebrow}
      </p>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function ModelIdentity({
  modelName,
  color,
  secondary,
}: {
  modelName: string;
  color: string;
  secondary?: string;
}) {
  return (
    <div className="model-identity">
      <span className="model-dot" style={{ backgroundColor: color }} />
      <div>
        <p className="model-identity-name">{modelName}</p>
        {secondary ? <p className="model-identity-secondary">{secondary}</p> : null}
      </div>
    </div>
  );
}

function ValueDot(props: { cx?: number; cy?: number; fill?: string }) {
  if (props.cx == null || props.cy == null) {
    return null;
  }
  return (
    <circle
      cx={props.cx}
      cy={props.cy}
      r={9}
      fill={props.fill ?? "#38bdf8"}
      stroke="rgba(10, 10, 15, 0.95)"
      strokeWidth={2}
    />
  );
}

function HeroCard({
  label,
  tooltip,
  color,
  heroValue,
  secondaryLine,
}: {
  label: string;
  tooltip: string;
  color: string;
  heroValue: string;
  secondaryLine: string;
}) {
  return (
    <article className="hero-card" style={{ borderTopColor: color }}>
      <p className="hero-card-label">
        {label}
        <InfoTooltip text={tooltip} />
      </p>
      <p className="hero-card-number" style={{ color }}>
        {heroValue}
      </p>
      <p className="hero-card-secondary">{secondaryLine}</p>
    </article>
  );
}

function SortableHeader({
  label,
  sortKey,
  activeKey,
  direction,
  onSort,
  tooltip,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  direction: SortDirection;
  onSort: (key: SortKey) => void;
  tooltip: string;
}) {
  return (
    <button type="button" className="leaderboard-sort" onClick={() => onSort(sortKey)}>
      <span>{label}</span>
      <InfoTooltip text={tooltip} />
      <span className="leaderboard-sort-arrow">
        {activeKey === sortKey ? (direction === "asc" ? "\u25b2" : "\u25bc") : ""}
      </span>
    </button>
  );
}

function AgreementHeatmap({
  models,
  matrix,
}: {
  models: string[];
  matrix: Record<string, Record<string, number>>;
}) {
  if (models.length === 0) {
    return null;
  }

  return (
    <section className="visual-card dashboard-section-card">
      <QuestionHeader
        eyebrow="Consensus"
        title="How often do models agree with each other?"
        description="Each square shows pairwise primary-action agreement across the full run."
        accentColor="#a855f7"
      />
      <div
        className="agreement-heatmap"
        style={{ gridTemplateColumns: `160px repeat(${models.length}, minmax(82px, 1fr))` }}
      >
        <div className="heatmap-corner" />
        {models.map((model) => (
          <div key={`column-${model}`} className="heatmap-axis">
            <ModelIdentity modelName={model} color={modelColor(model)} />
          </div>
        ))}
        {models.map((rowModel) => (
          <div key={`row-${rowModel}`} className="heatmap-row-group">
            <div className="heatmap-axis heatmap-axis-row">
              <ModelIdentity modelName={rowModel} color={modelColor(rowModel)} />
            </div>
            {models.map((columnModel) => {
              const value =
                rowModel === columnModel ? 1 : (matrix[rowModel]?.[columnModel] ?? 0);
              return (
                <div
                  key={`${rowModel}-${columnModel}`}
                  className={`heatmap-cell ${rowModel === columnModel ? "diagonal" : ""}`}
                  style={{ backgroundColor: heatmapColor(value) }}
                >
                  <InfoTooltip
                    text={`${rowModel} and ${columnModel} agreed on ${formatPercent(value)} of segments`}
                    label={formatPercent(value)}
                    className="heatmap-tooltip"
                  />
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
}

function buildCostDetails(
  run: RunPayload,
  rows: ModelView[],
  segmentCount: number,
  sweepData: SweepMetrics | null
): CostDetail[] {
  const preferredOrder = sweepData?.variants ?? [];
  return rows
    .map((row) => {
      const modelResults = run.results.filter((result) => result.model_name === row.model_name);
      const variantTotals = new Map<string, number>();
      for (const result of modelResults) {
        const label =
          result.extraction_label?.trim() || result.extraction_variant_id?.trim() || "default";
        variantTotals.set(label, (variantTotals.get(label) ?? 0) + (result.estimated_cost ?? 0));
      }
      const variants = [...variantTotals.entries()]
        .map(([label, total_cost]) => ({ label, total_cost }))
        .sort((left, right) => {
          const leftIndex = preferredOrder.indexOf(left.label);
          const rightIndex = preferredOrder.indexOf(right.label);
          if (leftIndex !== -1 || rightIndex !== -1) {
            return (
              (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) -
              (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex)
            );
          }
          return right.total_cost - left.total_cost;
        });

      const totalCost = modelResults.reduce((sum, result) => sum + (result.estimated_cost ?? 0), 0);
      return {
        model_name: row.model_name,
        color: row.color,
        total_cost: totalCost || row.total_cost,
        cost_per_segment: segmentCount > 0 ? totalCost / segmentCount : row.cost_per_segment,
        variants,
      };
    })
    .filter((entry) => entry.total_cost != null);
}

function variantRankingSignature(cells: SweepMetrics["cells"], variant: string): string {
  return cells
    .filter((cell) => cell.variant_label === variant)
    .sort(
      (left, right) =>
        right.parse_success_rate - left.parse_success_rate ||
        left.model_name.localeCompare(right.model_name)
    )
    .map((cell) => cell.model_name)
    .join("|");
}

export function DashboardVisuals({
  run,
  rows,
  costBreakdown,
  sweepData,
}: DashboardVisualsProps) {
  const accuracyAvailable = hasAccuracy(rows);
  const segmentCount = Math.max(run.segments.length, 1);
  const primaryMetric = metricLabel(accuracyAvailable);

  const models = useMemo<ModelView[]>(
    () =>
      rows.map((row) => {
        const agreement_pct = row.agreement == null ? null : row.agreement * 100;
        const accuracy_pct = row.accuracy == null ? null : row.accuracy * 100;
        const primaryScore = metricValue(row, accuracyAvailable);
        const cost_per_segment =
          row.total_cost == null ? null : Number((row.total_cost / segmentCount).toFixed(6));
        return {
          ...row,
          color: modelColor(row.model_name),
          agreement_pct,
          accuracy_pct,
          primary_score: primaryScore,
          primary_score_pct: primaryScore == null ? null : primaryScore * 100,
          cost_per_segment,
          value_score: agreementPerDollarScore(row, segmentCount),
        };
      }),
    [accuracyAvailable, rows, segmentCount]
  );

  const [sortKey, setSortKey] = useState<SortKey>(accuracyAvailable ? "accuracy" : "agreement");
  const [sortDirection, setSortDirection] = useState<SortDirection>(
    defaultDirectionFor(accuracyAvailable ? "accuracy" : "agreement")
  );
  const [showMoreColumns, setShowMoreColumns] = useState(false);
  const expanded = showMoreColumns;

  useEffect(() => {
    const nextDefault = accuracyAvailable ? "accuracy" : "agreement";
    if (sortKey === "accuracy" && !accuracyAvailable) {
      setSortKey(nextDefault);
      setSortDirection(defaultDirectionFor(nextDefault));
    }
  }, [accuracyAvailable, sortKey]);

  useEffect(() => {
    if (!showMoreColumns && (sortKey === "latency" || sortKey === "cost" || sortKey === "stability")) {
      const nextDefault = accuracyAvailable ? "accuracy" : "agreement";
      setSortKey(nextDefault);
      setSortDirection(defaultDirectionFor(nextDefault));
    }
  }, [accuracyAvailable, showMoreColumns, sortKey]);

  const primaryRankedRows = useMemo(
    () =>
      [...models].sort((left, right) => {
        const leftScore = accuracyAvailable ? left.accuracy : left.agreement;
        const rightScore = accuracyAvailable ? right.accuracy : right.agreement;
        if (leftScore != null && rightScore != null && leftScore !== rightScore) {
          return rightScore - leftScore;
        }
        if (left.parse_rate != null && right.parse_rate != null && left.parse_rate !== right.parse_rate) {
          return right.parse_rate - left.parse_rate;
        }
        const leftCombined = overallModelScore(left);
        const rightCombined = overallModelScore(right);
        if (leftCombined != null && rightCombined != null && leftCombined !== rightCombined) {
          return rightCombined - leftCombined;
        }
        return left.model_name.localeCompare(right.model_name);
      }),
    [accuracyAvailable, models]
  );

  const sortedRows = useMemo(
    () => [...models].sort((left, right) => compareRows(left, right, sortKey, sortDirection)),
    [models, sortDirection, sortKey]
  );

  const topRank = primaryRankedRows[0] ?? null;
  const bestValue = useMemo(() => {
    const candidate = bestValueModel(models, segmentCount);
    return candidate
      ? models.find((row) => row.model_name === candidate.model_name) ?? null
      : null;
  }, [models, segmentCount]);
  const fastest = useMemo(() => {
    const candidate = fastestModel(models);
    return candidate
      ? models.find((row) => row.model_name === candidate.model_name) ?? null
      : null;
  }, [models]);

  const agreementChartData = useMemo(
    () =>
      [...models]
        .filter((row) => row.agreement_pct != null)
        .sort(
          (left, right) =>
            (right.agreement_pct ?? 0) - (left.agreement_pct ?? 0) ||
            left.model_name.localeCompare(right.model_name)
        ),
    [models]
  );
  const latencyData = useMemo(
    () =>
      [...models]
        .filter((row) => row.avg_latency_ms != null)
        .sort(
          (left, right) =>
            (left.avg_latency_ms ?? Number.POSITIVE_INFINITY) -
              (right.avg_latency_ms ?? Number.POSITIVE_INFINITY) ||
            left.model_name.localeCompare(right.model_name)
        ),
    [models]
  );
  const latencyDomain = domainWithPadding(
    latencyData
      .map((row) => row.avg_latency_ms)
      .filter((value): value is number => value != null),
    100
  );

  const scatterData = useMemo(
    () =>
      models
        .filter(
          (row): row is ModelView & { primary_score_pct: number; cost_per_segment: number } =>
            row.primary_score_pct != null && row.cost_per_segment != null
        )
        .map((row) => ({
          ...row,
          primary_score_pct: row.primary_score_pct,
          cost_per_segment: row.cost_per_segment,
        })),
    [models]
  );
  const scatterXDomain = domainWithPadding(scatterData.map((row) => row.primary_score_pct), 3);
  const scatterYDomain = domainWithPadding(
    scatterData.map((row) => row.cost_per_segment),
    0.00025
  );
  const scatterXMedian = median(scatterData.map((row) => row.primary_score_pct));
  const scatterYMedian = median(scatterData.map((row) => row.cost_per_segment));

  const costDetails = useMemo(
    () => buildCostDetails(run, models, segmentCount, sweepData),
    [models, run, segmentCount, sweepData]
  );

  const variantCharts = useMemo(() => {
    if (!sweepData) {
      return [];
    }
    return sweepData.variants.map((variant) => {
      const cells = sweepData.cells
        .filter((cell) => cell.variant_label === variant)
        .sort(
          (left, right) =>
            right.parse_success_rate - left.parse_success_rate ||
            left.model_name.localeCompare(right.model_name)
        )
        .map((cell) => ({
          ...cell,
          color: modelColor(cell.model_name),
          parse_success_pct: cell.parse_success_rate * 100,
        }));
      const agreementMatrix = sweepData.agreement_by_variant[variant] ?? {};
      const offDiagonalValues = Object.entries(agreementMatrix).flatMap(([rowModel, row]) =>
        Object.entries(row)
          .filter(([columnModel]) => columnModel !== rowModel)
          .map(([, value]) => value)
      );
      const agreementAverage =
        offDiagonalValues.length > 0
          ? offDiagonalValues.reduce((sum, value) => sum + value, 0) / offDiagonalValues.length
          : null;
      return {
        variant,
        cells,
        rankingSignature: variantRankingSignature(sweepData.cells, variant),
        agreementAverage,
      };
    });
  }, [sweepData]);

  const baselineSignature = variantCharts[0]?.rankingSignature ?? null;
  const rankingShifts = new Set(
    variantCharts
      .filter(
        (variant) => baselineSignature != null && variant.rankingSignature !== baselineSignature
      )
      .map((variant) => variant.variant)
  );

  const totalCost =
    costBreakdown?.total_cost ??
    run.results.reduce((sum, result) => sum + (result.estimated_cost ?? 0), 0);

  const agreementExtremes = metricExtremes(models, "agreement");
  function handleSort(nextKey: SortKey): void {
    if (nextKey === sortKey) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(nextKey);
    setSortDirection(defaultDirectionFor(nextKey));
  }

  return (
    <div className="dashboard-section-stack">
      <section className="hero-grid">
        <HeroCard
          label="Best Overall"
          tooltip={
            accuracyAvailable
              ? "Based on accuracy score. Ground truth is available for this run, so this reflects verified correctness."
              : "Based on agreement score. This is model consensus, not verified accuracy. Provide ground_truth.json for real accuracy measurement."
          }
          color={topRank?.color ?? "#4c9aff"}
          heroValue={topRank?.primary_score != null ? formatPercent(topRank.primary_score) : "\u2014"}
          secondaryLine={
            topRank?.primary_score != null
              ? `${topRank.model_name} \u00b7 ${formatMoney(topRank.cost_per_segment)} / segment`
              : "Not enough data"
          }
        />
        <HeroCard
          label="Best Value"
          tooltip={`${accuracyAvailable ? "Accuracy" : "Agreement"} divided by cost per segment. Higher = more insight per dollar.`}
          color={bestValue ? modelColor(bestValue.model_name) : "#22c55e"}
          heroValue={
            bestValue?.total_cost === 0
              ? "\u221e"
              : bestValue?.value_score != null && Number.isFinite(bestValue.value_score)
                ? bestValue.value_score.toFixed(1)
                : "\u2014"
          }
          secondaryLine={
            bestValue?.total_cost === 0 ||
            (bestValue?.value_score != null && Number.isFinite(bestValue.value_score))
              ? `${bestValue.model_name} \u00b7 ${formatPercent(bestValue.primary_score)} ${primaryMetric}`
              : "Not enough data"
          }
        />
        <HeroCard
          label="Fastest"
          tooltip="Average response latency across all segments and variants."
          color={fastest ? modelColor(fastest.model_name) : "#f59e0b"}
          heroValue={
            fastest?.avg_latency_ms != null && fastest.avg_latency_ms > 0
              ? formatLatency(fastest.avg_latency_ms)
              : "\u2014"
          }
          secondaryLine={
            fastest?.avg_latency_ms != null && fastest.avg_latency_ms > 0
              ? `${fastest.model_name} \u00b7 ${formatPercent(fastest.parse_rate)} parse`
              : "Not enough data"
          }
        />
      </section>

      <section className="visual-card dashboard-section-card">
        <QuestionHeader
          eyebrow="Leaderboard"
          title={
            accuracyAvailable
              ? "Which model is most accurate on this run?"
              : "Which model leads the consensus?"
          }
          description="Sort by the metric that matters to you. The leaderboard defaults to verified accuracy when ground truth exists, otherwise model agreement."
          accentColor="#4c9aff"
        />

        <div className="leaderboard-toolbar">
          <button
            type="button"
            className="ghost-btn small"
            onClick={() => setShowMoreColumns((current) => !current)}
          >
            {showMoreColumns ? "Fewer columns" : "More columns"}
          </button>
        </div>

        {!accuracyAvailable ? (
          <div className="info-banner">
            <span className="info-banner-icon">i</span>
            <div>
              <strong>These results show model agreement (consensus), not verified accuracy.</strong>
              <p>To measure real accuracy, run with --ground-truth labels.json.</p>
            </div>
          </div>
        ) : null}

        <div className="leaderboard-scroll">
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>
                  <SortableHeader
                    label="Model"
                    sortKey="model"
                    activeKey={sortKey}
                    direction={sortDirection}
                    onSort={handleSort}
                    tooltip="Model identifier for this run."
                  />
                </th>
                <th>
                  <SortableHeader
                    label="Agreement"
                    sortKey="agreement"
                    activeKey={sortKey}
                    direction={sortDirection}
                    onSort={handleSort}
                    tooltip="How often this model gives the same action label as other models. High agreement means models converge on the same answer - but doesn't guarantee correctness."
                  />
                </th>
                <th>
                  <SortableHeader
                    label="Accuracy"
                    sortKey="accuracy"
                    activeKey={sortKey}
                    direction={sortDirection}
                    onSort={handleSort}
                    tooltip="Exact match rate against ground truth, if available. No ground truth labels provided means the dashboard cannot measure real accuracy."
                  />
                </th>
                <th>
                  <SortableHeader
                    label="Confidence"
                    sortKey="confidence"
                    activeKey={sortKey}
                    direction={sortDirection}
                    onSort={handleSort}
                    tooltip="The model's self-reported certainty (0-1). Models can be confidently wrong."
                  />
                </th>
                {expanded ? (
                  <th>
                    <SortableHeader
                      label="Latency"
                      sortKey="latency"
                      activeKey={sortKey}
                      direction={sortDirection}
                      onSort={handleSort}
                      tooltip="Average time for the model to respond, in milliseconds."
                    />
                  </th>
                ) : null}
                {expanded ? (
                  <th>
                    <SortableHeader
                      label="Cost / segment"
                      sortKey="cost"
                      activeKey={sortKey}
                      direction={sortDirection}
                      onSort={handleSort}
                      tooltip="Estimated API cost per video segment based on token usage."
                    />
                  </th>
                ) : null}
                {expanded && sweepData ? (
                  <th>
                    <SortableHeader
                      label="Stability"
                      sortKey="stability"
                      activeKey={sortKey}
                      direction={sortDirection}
                      onSort={handleSort}
                      tooltip="Does this model give the same answer when you change the number of input frames? Higher = more stable."
                    />
                  </th>
                ) : null}
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row, index) => (
                <tr
                  key={row.model_name}
                  className={`leaderboard-row ${index === 0 ? "top-ranked" : ""}`}
                  style={{ boxShadow: `inset 4px 0 0 ${row.color}` }}
                >
                  <td className="leaderboard-rank">#{index + 1}</td>
                  <td>
                    <ModelIdentity
                      modelName={row.model_name}
                      color={row.color}
                      secondary={
                        index === 0
                          ? accuracyAvailable
                            ? "Highest verified accuracy"
                            : "Highest model agreement"
                          : undefined
                      }
                    />
                  </td>
                  <td className={cellClass(row.agreement, agreementExtremes.best, agreementExtremes.worst)}>
                    {formatPercent(row.agreement)}
                  </td>
                  <td>
                    {row.accuracy == null ? (
                      <span className="missing-metric">
                        -
                        <InfoTooltip text="No ground truth labels provided. Add a ground_truth.json file to measure real accuracy." />
                      </span>
                    ) : (
                      formatPercent(row.accuracy)
                    )}
                  </td>
                  <td>{formatConfidence(row.confidence)}</td>
                  {expanded ? (
                    <td>{formatLatency(row.avg_latency_ms)}</td>
                  ) : null}
                  {expanded ? (
                    <td>{formatMoney(row.cost_per_segment)}</td>
                  ) : null}
                  {expanded && sweepData ? (
                    <td>{row.stability == null ? "-" : row.stability.toFixed(2)}</td>
                  ) : null}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="secondary-chart-block">
          <QuestionHeader
            eyebrow="Visual Ranking"
            title="How does agreement stack up at a glance?"
            description="The bar chart stays visible as a quick visual ranking, even when you sort the table by another metric."
            accentColor="#22c55e"
          />
          <div className="chart-stage">
            <ResponsiveContainer
              width="100%"
              height={Math.min(200, Math.max(136, agreementChartData.length * 30 + 24))}
            >
              <BarChart
                data={agreementChartData}
                layout="vertical"
                margin={{ top: 6, right: 20, left: 12, bottom: 6 }}
                barCategoryGap={12}
              >
                <CartesianGrid horizontal={false} stroke="rgba(148, 163, 184, 0.1)" />
                <XAxis
                  type="number"
                  domain={[0, 50]}
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(value) => `${Math.round(numericValue(value))}%`}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="model_name"
                  width={170}
                  tick={{ fill: "#e2e8f0", fontSize: 14 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(76, 154, 255, 0.08)" }}
                  contentStyle={tooltipStyle()}
                  formatter={(value) => [
                    `${Math.round(numericValue(value))}% agreement`,
                    "Agreement",
                  ]}
                />
                <Bar dataKey="agreement_pct" radius={[0, 14, 14, 0]} barSize={32}>
                  {agreementChartData.map((row) => (
                    <Cell key={row.model_name} fill={row.color} />
                  ))}
                  <LabelList
                    dataKey="agreement_pct"
                    position="right"
                    formatter={(value) =>
                      `${Math.round(numericValue(value))}%`
                    }
                    fill="#f8fafc"
                    fontSize={16}
                    fontWeight={700}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="visual-card dashboard-section-card">
        <QuestionHeader
          eyebrow="Value"
          title="Which model buys the most signal per dollar?"
          description={`The best place to be is bottom-right: high ${primaryMetric}, low cost per segment.`}
          accentColor="#f59e0b"
        />
        <div className="chart-stage-large">
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 18, right: 36, left: 24, bottom: 18 }}>
              <CartesianGrid stroke="rgba(148, 163, 184, 0.1)" />
              <XAxis
                type="number"
                dataKey="primary_score_pct"
                name={primaryMetric}
                domain={scatterXDomain ?? [0, 100]}
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                tickFormatter={(value) => `${Math.round(numericValue(value))}%`}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="number"
                dataKey="cost_per_segment"
                name="Cost per segment"
                domain={scatterYDomain ?? [0, 0.01]}
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                tickFormatter={(value) => formatMoney(numericValue(value))}
                axisLine={false}
                tickLine={false}
                width={92}
              />
              {scatterXMedian != null ? (
                <ReferenceLine
                  x={scatterXMedian}
                  stroke="rgba(148, 163, 184, 0.22)"
                  strokeDasharray="4 4"
                />
              ) : null}
              {scatterYMedian != null ? (
                <ReferenceLine
                  y={scatterYMedian}
                  stroke="rgba(148, 163, 184, 0.22)"
                  strokeDasharray="4 4"
                />
              ) : null}
              <Tooltip
                cursor={{ strokeDasharray: "4 4" }}
                contentStyle={tooltipStyle()}
                formatter={(value, name) => {
                  if (name === "cost_per_segment") {
                    return [formatMoney(numericValue(value)), "Cost / segment"];
                  }
                  return [`${Math.round(numericValue(value))}%`, primaryMetric];
                }}
              />
              <Scatter data={scatterData} shape={<ValueDot />}>
                {scatterData.map((row) => (
                  <Cell key={row.model_name} fill={row.color} />
                ))}
                <LabelList dataKey="model_name" position="top" fill="#cbd5e1" fontSize={12} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="cost-detail-stack">
          {costDetails.map((detail) => (
            <details key={detail.model_name} className="cost-detail-card">
              <summary>
                <ModelIdentity
                  modelName={detail.model_name}
                  color={detail.color}
                  secondary="Cost breakdown"
                />
                <div className="cost-detail-summary">
                  <span>Total {formatMoney(detail.total_cost)}</span>
                  <span>Per segment {formatMoney(detail.cost_per_segment)}</span>
                </div>
              </summary>
              <div className="cost-detail-body">
                <div className="mini-stat-grid">
                  <div>
                    <span>Total cost</span>
                    <strong>{formatMoney(detail.total_cost)}</strong>
                  </div>
                  <div>
                    <span>Cost / segment</span>
                    <strong>{formatMoney(detail.cost_per_segment)}</strong>
                  </div>
                  <div>
                    <span>Run total</span>
                    <strong>{formatMoney(totalCost)}</strong>
                  </div>
                  <div>
                    <span>Segments</span>
                    <strong>{segmentCount}</strong>
                  </div>
                </div>
                {detail.variants.length > 1 ? (
                  <div className="variant-chip-grid">
                    {detail.variants.map((variant) => (
                      <div key={`${detail.model_name}-${variant.label}`} className="metric-chip">
                        <span>{variant.label}</span>
                        <strong>{formatMoney(variant.total_cost)}</strong>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            </details>
          ))}
        </div>
      </section>

      <section className="visual-card dashboard-section-card">
        <QuestionHeader
          eyebrow="Latency"
          title="Which model responds the fastest?"
          description="Shorter bars mean quicker responses. Color shifts from green to amber to red as models slow down."
          accentColor="#22c55e"
        />
        <div className="chart-stage">
          <ResponsiveContainer width="100%" height={Math.max(240, latencyData.length * 44 + 28)}>
            <BarChart
              data={latencyData}
              layout="vertical"
              margin={{ top: 6, right: 20, left: 12, bottom: 6 }}
              barCategoryGap={12}
            >
              <CartesianGrid horizontal={false} stroke="rgba(148, 163, 184, 0.1)" />
              <XAxis
                type="number"
                domain={latencyDomain ?? [0, 1000]}
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                tickFormatter={(value) => `${Math.round(numericValue(value))} ms`}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="model_name"
                width={170}
                tick={{ fill: "#e2e8f0", fontSize: 14 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(76, 154, 255, 0.08)" }}
                contentStyle={tooltipStyle()}
                formatter={(value) => [
                  `${Math.round(numericValue(value))} ms`,
                  "Latency",
                ]}
              />
              <Bar dataKey="avg_latency_ms" radius={[0, 14, 14, 0]} barSize={32}>
                {latencyData.map((row, index) => (
                  <Cell key={row.model_name} fill={latencyColor(index, latencyData.length)} />
                ))}
                <LabelList
                  dataKey="avg_latency_ms"
                  position="right"
                  formatter={(value) =>
                    `${Math.round(numericValue(value))} ms`
                  }
                  fill="#f8fafc"
                  fontSize={16}
                  fontWeight={700}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <AgreementHeatmap models={run.models} matrix={run.agreement} />

      {sweepData ? (
        <details className="visual-card dashboard-section-card sweep-card">
          <summary className="sweep-summary">
            <div>
              <QuestionHeader
                eyebrow="Sweep"
                title="Does frame count change the ranking?"
                description="Each mini chart shows parse success by extraction variant. Amber badges flag variants where the model order changes."
                accentColor="#f59e0b"
              />
            </div>
            {rankingShifts.size > 0 ? (
              <span className="warning-pill">{rankingShifts.size} variants shift the ranking</span>
            ) : (
              <span className="status-chip live">Ranking is stable</span>
            )}
          </summary>

          <div className="stability-strip">
            {sweepData.stability.map((entry) => (
              <div key={entry.model_name} className="stability-pill">
                <span>{entry.model_name}</span>
                <strong>{formatPercent(entry.self_agreement)}</strong>
                <p className="hero-card-secondary">rank stability {entry.rank_stability.toFixed(2)}</p>
              </div>
            ))}
          </div>

          <div className="sweep-mini-grid">
            {variantCharts.map((variant) => (
              <article key={variant.variant} className="mini-variant-card">
                <div className="mini-variant-head">
                  <div>
                    <h4>{variant.variant}</h4>
                    <p className="hero-card-secondary">
                      mean pairwise agreement {variant.agreementAverage == null ? "-" : formatPercent(variant.agreementAverage)}
                    </p>
                  </div>
                  {rankingShifts.has(variant.variant) ? (
                    <span className="warning-pill">ranking changed</span>
                  ) : null}
                </div>
                <ResponsiveContainer width="100%" height={variant.cells.length * 40 + 24}>
                  <BarChart
                    data={variant.cells}
                    layout="vertical"
                    margin={{ top: 8, right: 14, left: 0, bottom: 0 }}
                    barCategoryGap={10}
                  >
                    <XAxis type="number" domain={[0, 100]} hide />
                    <YAxis
                      type="category"
                      dataKey="model_name"
                      width={132}
                      tick={{ fill: "#cbd5e1", fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Bar dataKey="parse_success_pct" radius={[0, 12, 12, 0]} barSize={24}>
                      {variant.cells.map((cell) => (
                        <Cell key={`${variant.variant}-${cell.model_name}`} fill={cell.color} />
                      ))}
                      <LabelList
                        dataKey="parse_success_pct"
                        position="right"
                        formatter={(value) =>
                          `${Math.round(numericValue(value))}%`
                        }
                        fill="#f8fafc"
                        fontSize={13}
                        fontWeight={600}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </article>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}
