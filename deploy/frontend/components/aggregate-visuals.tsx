"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  formatLatency,
  formatMoney,
  formatPercent,
  modelColor,
} from "../lib/analysis";
import type { AggregateModelRow } from "./aggregate-dashboard";

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

function ScatterPoint(props: { cx?: number; cy?: number; fill?: string }) {
  if (props.cx == null || props.cy == null) {
    return null;
  }
  return (
    <circle
      cx={props.cx}
      cy={props.cy}
      r={8}
      fill={props.fill ?? "#38bdf8"}
      stroke="rgba(10, 10, 15, 0.95)"
      strokeWidth={2}
    />
  );
}

export function AggregateVisuals({ rows }: { rows: AggregateModelRow[] }) {
  const agreementData = [...rows]
    .filter((row) => row.avg_agreement != null)
    .map((row) => ({
      ...row,
      agreement_pct: (row.avg_agreement ?? 0) * 100,
      fill: modelColor(row.model_name),
    }))
    .sort(
      (left, right) =>
        right.agreement_pct - left.agreement_pct || left.model_name.localeCompare(right.model_name)
    );

  const accuracyAgreementData = [...rows]
    .filter((row) => row.avg_agreement != null || row.avg_accuracy != null)
    .map((row) => ({
      ...row,
      agreement_pct: row.avg_agreement == null ? null : row.avg_agreement * 100,
      accuracy_pct: row.avg_accuracy == null ? null : row.avg_accuracy * 100,
    }));

  const latencyData = [...rows]
    .filter((row) => row.avg_latency_ms != null)
    .map((row) => ({
      ...row,
      fill: modelColor(row.model_name),
    }))
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

  const scatterData = rows
    .filter(
      (row): row is AggregateModelRow & { avg_accuracy: number } =>
        row.avg_accuracy != null && row.total_cost > 0
    )
    .map((row) => ({
      ...row,
      accuracy_pct: row.avg_accuracy * 100,
      fill: modelColor(row.model_name),
    }));

  const scatterXDomain = domainWithPadding(
    scatterData.map((row) => row.total_cost),
    0.01
  );
  const scatterYDomain = domainWithPadding(
    scatterData.map((row) => row.accuracy_pct),
    4
  );
  const averageAgreement = average(
    rows.map((row) => row.avg_agreement).filter((value): value is number => value != null)
  );
  const averageLatency = average(
    rows.map((row) => row.avg_latency_ms).filter((value): value is number => value != null)
  );

  return (
    <div className="aggregate-visual-stack">
      <section className="visual-card dashboard-section-card">
        <SectionHeader
          eyebrow="Agreement Comparison"
          title="Which models attract the strongest consensus?"
          description="Average agreement across the loaded run history, with each bar colored by model identity."
        />

        {agreementData.length === 0 ? (
          <p className="empty-state">No agreement data is available for the loaded runs.</p>
        ) : (
          <div className="chart-stage-large">
            <ResponsiveContainer
              width="100%"
              height={Math.max(220, agreementData.length * 52)}
            >
              <BarChart
                data={agreementData}
                layout="vertical"
                margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
                barCategoryGap={14}
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
                  width={160}
                  tick={{ fill: "#f8fafc", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={tooltipStyle()}
                  cursor={{ fill: "rgba(148, 163, 184, 0.08)" }}
                  formatter={(value) => [`${numericValue(value).toFixed(1)}%`, "Avg agreement"]}
                />
                <Bar dataKey="agreement_pct" radius={[0, 12, 12, 0]}>
                  {agreementData.map((entry) => (
                    <Cell key={entry.model_name} fill={entry.fill} fillOpacity={0.9} />
                  ))}
                  <LabelList
                    dataKey="agreement_pct"
                    position="right"
                    formatter={(value: unknown) => `${Math.round(numericValue(value))}%`}
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

      <div className="analysis-grid two-up">
        <section className="visual-card dashboard-section-card">
          <SectionHeader
            eyebrow="Consensus vs Correctness"
            title="Where does agreement diverge from accuracy?"
            description="Two bars per model: blue is agreement, green is accuracy. Big gaps highlight quietly-correct or confidently-wrong models."
          />

          {accuracyAgreementData.length === 0 ? (
            <p className="empty-state">Not enough metric data is available for this comparison yet.</p>
          ) : (
            <div className="chart-stage">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart
                  data={accuracyAgreementData}
                  margin={{ top: 8, right: 16, left: 0, bottom: 16 }}
                  barGap={6}
                  barCategoryGap="22%"
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis
                    dataKey="model_name"
                    tick={{ fill: "#cbd5e1", fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    interval={0}
                    angle={-18}
                    textAnchor="end"
                    height={68}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) => `${Math.round(numericValue(value))}%`}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle()}
                    formatter={(value, name) => [
                      `${numericValue(value).toFixed(1)}%`,
                      name === "agreement_pct" ? "Avg agreement" : "Avg accuracy",
                    ]}
                  />
                  <Bar
                    dataKey="agreement_pct"
                    name="agreement_pct"
                    fill="#4c9aff"
                    radius={[8, 8, 0, 0]}
                  />
                  <Bar
                    dataKey="accuracy_pct"
                    name="accuracy_pct"
                    fill="#22c55e"
                    radius={[8, 8, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

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
              <ResponsiveContainer
                width="100%"
                height={Math.max(240, latencyData.length * 52)}
              >
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

          {averageLatency != null ? (
            <p className="table-note">
              Average latency across loaded models is {formatLatency(averageLatency)}.
            </p>
          ) : null}
        </section>
      </div>

      {scatterData.length > 1 ? (
        <section className="visual-card dashboard-section-card">
          <SectionHeader
            eyebrow="Cost Efficiency"
            title="Who delivers the best accuracy per dollar?"
            description="Each dot is a model. Up is better accuracy, left is lower cost."
          />

          <div className="chart-stage-large">
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 12, right: 36, left: 12, bottom: 18 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                <XAxis
                  type="number"
                  dataKey="total_cost"
                  domain={scatterXDomain ?? [0, 1]}
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(value) => `$${numericValue(value).toFixed(2)}`}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="number"
                  dataKey="accuracy_pct"
                  domain={scatterYDomain ?? [0, 100]}
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(value) => `${Math.round(numericValue(value))}%`}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={tooltipStyle()}
                  formatter={(value, name) => [
                    name === "total_cost"
                      ? formatMoney(numericValue(value))
                      : `${numericValue(value).toFixed(1)}%`,
                    name === "total_cost" ? "Total cost" : "Avg accuracy",
                  ]}
                  labelFormatter={(_, payload) =>
                    payload?.[0] && "payload" in payload[0] && payload[0].payload
                      ? String((payload[0].payload as { model_name: string }).model_name)
                      : ""
                  }
                />
                <Scatter data={scatterData} shape={<ScatterPoint />}>
                  <LabelList dataKey="model_name" position="top" fill="#e2e8f0" fontSize={11} />
                  {scatterData.map((entry) => (
                    <Cell key={entry.model_name} fill={entry.fill} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>
      ) : null}
    </div>
  );
}
