"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatLatency, formatPercent } from "../lib/analysis";
import type { AggregateModelRow } from "./aggregate-dashboard";

type ChartDatum = AggregateModelRow & {
  agreement_pct: number | null;
  primary_accuracy_pct: number | null;
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

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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

function SectionHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-heading">
      <p className="section-eyebrow">{eyebrow}</p>
      <h3>{title}</h3>
      <p className="chart-desc">{description}</p>
    </div>
  );
}

function primaryAccuracy(row: AggregateModelRow): number | null {
  return row.avg_llm_accuracy ?? row.avg_accuracy;
}

export function AggregateVisuals({ rows }: { rows: AggregateModelRow[] }) {
  const chartRows: ChartDatum[] = rows.map((row) => ({
    ...row,
    agreement_pct: row.avg_agreement == null ? null : row.avg_agreement * 100,
    primary_accuracy_pct: primaryAccuracy(row) == null ? null : primaryAccuracy(row)! * 100,
  }));

  const accuracyRankingData = [...chartRows]
    .filter(
      (row): row is ChartDatum & { primary_accuracy_pct: number } => row.primary_accuracy_pct != null
    )
    .sort(
      (left, right) =>
        right.primary_accuracy_pct - left.primary_accuracy_pct ||
        (right.agreement_pct ?? 0) - (left.agreement_pct ?? 0) ||
        left.model_name.localeCompare(right.model_name)
    );

  const agreementRankingData = [...chartRows]
    .filter((row): row is ChartDatum & { agreement_pct: number } => row.agreement_pct != null)
    .sort(
      (left, right) =>
        right.agreement_pct - left.agreement_pct ||
        (right.primary_accuracy_pct ?? 0) - (left.primary_accuracy_pct ?? 0) ||
        left.model_name.localeCompare(right.model_name)
    );

  const latencyData = [...chartRows]
    .filter((row) => row.avg_latency_ms != null)
    .sort(
      (left, right) =>
        (left.avg_latency_ms ?? Number.POSITIVE_INFINITY) -
          (right.avg_latency_ms ?? Number.POSITIVE_INFINITY) ||
        left.model_name.localeCompare(right.model_name)
    );

  const latencyDomain = domainWithPadding(
    latencyData
      .map((row) => row.avg_latency_ms)
      .filter((value): value is number => value != null),
    100
  );
  const averageAgreement = average(
    rows.map((row) => row.avg_agreement).filter((value): value is number => value != null)
  );

  return (
    <div className="aggregate-visual-stack">
      <div className="dual-chart-row">
        <section className="visual-card dashboard-section-card">
          <SectionHeader
            eyebrow="Ranking"
            title="ACCURACY RANKING"
            description="Which models get the right answer?"
          />

          {accuracyRankingData.length === 0 ? (
            <p className="empty-state ranking-chart-placeholder">
              Run a benchmark with --ground-truth to see accuracy rankings
            </p>
          ) : (
            <div className="chart-stage-large">
              <ResponsiveContainer
                width="100%"
                height={Math.max(240, accuracyRankingData.length * 60)}
              >
                <BarChart
                  data={accuracyRankingData}
                  layout="vertical"
                  margin={{ top: 8, right: 28, left: 8, bottom: 8 }}
                  barCategoryGap={18}
                >
                  <CartesianGrid horizontal={false} stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis
                    type="number"
                    domain={[0, 100]}
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    tickFormatter={(value) => `${Math.round(numericValue(value))}%`}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    dataKey="model_name"
                    type="category"
                    width={170}
                    tick={{ fill: "#f8fafc", fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle()}
                    cursor={{ fill: "rgba(148, 163, 184, 0.08)" }}
                    formatter={(value) => [`${numericValue(value).toFixed(1)}%`, "Accuracy"]}
                  />
                  <Bar dataKey="primary_accuracy_pct" fill="#22c55e" fillOpacity={0.92} radius={[0, 12, 12, 0]}>
                    <LabelList
                      dataKey="primary_accuracy_pct"
                      position="right"
                      formatter={(value: unknown) =>
                        value == null ? "" : `${Math.round(numericValue(value))}%`
                      }
                      fill="#f8fafc"
                      fontSize={12}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        <section className="visual-card dashboard-section-card">
          <SectionHeader
            eyebrow="Ranking"
            title="AGREEMENT RANKING"
            description="Which models agree with each other?"
          />

          {agreementRankingData.length === 0 ? (
            <p className="empty-state">No agreement data is available for the loaded models.</p>
          ) : (
            <div className="chart-stage-large">
              <ResponsiveContainer
                width="100%"
                height={Math.max(240, agreementRankingData.length * 60)}
              >
                <BarChart
                  data={agreementRankingData}
                  layout="vertical"
                  margin={{ top: 8, right: 28, left: 8, bottom: 8 }}
                  barCategoryGap={18}
                >
                  <CartesianGrid horizontal={false} stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis
                    type="number"
                    domain={[0, 100]}
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    tickFormatter={(value) => `${Math.round(numericValue(value))}%`}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    dataKey="model_name"
                    type="category"
                    width={170}
                    tick={{ fill: "#f8fafc", fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle()}
                    cursor={{ fill: "rgba(148, 163, 184, 0.08)" }}
                    formatter={(value) => [`${numericValue(value).toFixed(1)}%`, "Agreement"]}
                  />
                  <Bar dataKey="agreement_pct" fill="#4c9aff" fillOpacity={0.92} radius={[0, 12, 12, 0]}>
                    <LabelList
                      dataKey="agreement_pct"
                      position="right"
                      formatter={(value: unknown) =>
                        value == null ? "" : `${Math.round(numericValue(value))}%`
                      }
                      fill="#f8fafc"
                      fontSize={12}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {averageAgreement != null ? (
            <p className="table-note">
              Loaded-run average agreement sits at {formatPercent(averageAgreement)} across{" "}
              {rows.length} models.
            </p>
          ) : null}
        </section>
      </div>

      <section className="visual-card dashboard-section-card">
        <SectionHeader
          eyebrow="Latency Breakdown"
          title="Which models stay fast across runs?"
          description="A horizontal latency ranking, colored from fast green through amber to slow red."
        />

        {latencyData.length === 0 ? (
          <p className="empty-state">No latency data is available for the loaded runs.</p>
        ) : (
          <div className="chart-stage">
            <ResponsiveContainer width="100%" height={Math.max(240, latencyData.length * 52)}>
              <BarChart
                data={latencyData}
                layout="vertical"
                margin={{ top: 8, right: 28, left: 8, bottom: 8 }}
                barCategoryGap={14}
              >
                <CartesianGrid horizontal={false} stroke="rgba(148, 163, 184, 0.1)" />
                <XAxis
                  type="number"
                  domain={latencyDomain ?? [0, 1000]}
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(value) => `${Math.round(numericValue(value))}ms`}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  dataKey="model_name"
                  type="category"
                  width={160}
                  tick={{ fill: "#f8fafc", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={tooltipStyle()}
                  formatter={(value) => [formatLatency(numericValue(value)), "Avg latency"]}
                />
                <Bar dataKey="avg_latency_ms" radius={[0, 12, 12, 0]}>
                  {latencyData.map((entry, index) => (
                    <Cell
                      key={entry.model_name}
                      fill={latencyColor(index, latencyData.length)}
                      fillOpacity={0.9}
                    />
                  ))}
                  <LabelList
                    dataKey="avg_latency_ms"
                    position="right"
                    formatter={(value: unknown) => `${Math.round(numericValue(value))}ms`}
                    fill="#f8fafc"
                    fontSize={12}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>
    </div>
  );
}
