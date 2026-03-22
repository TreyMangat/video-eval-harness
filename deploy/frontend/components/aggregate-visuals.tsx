"use client";

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

import { formatLatency, formatMoney, formatPercent, modelColor } from "../lib/analysis";
import type { AggregateModelRow } from "./aggregate-dashboard";

type ChartDatum = AggregateModelRow & {
  agreement_pct: number | null;
  exact_accuracy_pct: number | null;
  llm_accuracy_pct: number | null;
  primary_accuracy_pct: number | null;
  fill: string;
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

function primaryAccuracy(row: AggregateModelRow): number | null {
  return row.avg_llm_accuracy ?? row.avg_accuracy;
}

export function AggregateVisuals({ rows }: { rows: AggregateModelRow[] }) {
  const chartRows: ChartDatum[] = rows.map((row) => ({
    ...row,
    agreement_pct: row.avg_agreement == null ? null : row.avg_agreement * 100,
    exact_accuracy_pct: row.avg_accuracy == null ? null : row.avg_accuracy * 100,
    llm_accuracy_pct: row.avg_llm_accuracy == null ? null : row.avg_llm_accuracy * 100,
    primary_accuracy_pct: primaryAccuracy(row) == null ? null : primaryAccuracy(row)! * 100,
    fill: modelColor(row.model_name),
  }));

  const accuracyAgreementScatterData = [...chartRows]
    .filter(
      (row): row is ChartDatum & { agreement_pct: number; llm_accuracy_pct: number } =>
        row.agreement_pct != null && row.llm_accuracy_pct != null
    )
    .sort(
      (left, right) =>
        right.llm_accuracy_pct - left.llm_accuracy_pct ||
        right.agreement_pct - left.agreement_pct ||
        left.model_name.localeCompare(right.model_name)
    );

  const agreementComparisonData = [...chartRows]
    .filter((row) => row.agreement_pct != null || row.llm_accuracy_pct != null)
    .sort(
      (left, right) =>
        (right.llm_accuracy_pct ?? right.agreement_pct ?? 0) -
          (left.llm_accuracy_pct ?? left.agreement_pct ?? 0) ||
        (right.agreement_pct ?? 0) - (left.agreement_pct ?? 0) ||
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

  const costScatterData = [...chartRows]
    .filter(
      (row): row is ChartDatum & { primary_accuracy_pct: number } =>
        row.primary_accuracy_pct != null && row.total_cost > 0
    )
    .sort(
      (left, right) =>
        right.primary_accuracy_pct - left.primary_accuracy_pct ||
        left.total_cost - right.total_cost ||
        left.model_name.localeCompare(right.model_name)
    );

  const latencyDomain = domainWithPadding(
    latencyData
      .map((row) => row.avg_latency_ms)
      .filter((value): value is number => value != null),
    100
  );
  const costScatterXDomain = domainWithPadding(
    costScatterData.map((row) => row.total_cost),
    0.01
  );
  const costScatterYDomain = domainWithPadding(
    costScatterData.map((row) => row.primary_accuracy_pct),
    4
  );

  const averageAgreement = average(
    rows.map((row) => row.avg_agreement).filter((value): value is number => value != null)
  );
  const averageAccuracyGap = average(
    accuracyAgreementScatterData.map((row) => Math.abs(row.llm_accuracy_pct - row.agreement_pct))
  );

  return (
    <div className="aggregate-visual-stack">
      <section className="visual-card dashboard-section-card">
        <SectionHeader
          eyebrow="Transparency"
          title="AGREEMENT VS ACCURACY"
          description="Models above the diagonal are quietly correct. Models below are confidently wrong."
        />

        {accuracyAgreementScatterData.length === 0 ? (
          <p className="empty-state">
            No accuracy data is available in the loaded runs yet. Add a ground-truth or judge-scored
            run to compare consensus against correctness.
          </p>
        ) : (
          <div className="chart-stage-large">
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 16, right: 32, left: 16, bottom: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                <XAxis
                  type="number"
                  dataKey="agreement_pct"
                  domain={[0, 100]}
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(value) => `${Math.round(numericValue(value))}%`}
                  axisLine={false}
                  tickLine={false}
                  label={{
                    value: "Average agreement",
                    position: "insideBottom",
                    offset: -10,
                    fill: "#94a3b8",
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="llm_accuracy_pct"
                  domain={[0, 100]}
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(value) => `${Math.round(numericValue(value))}%`}
                  axisLine={false}
                  tickLine={false}
                  label={{
                    value: "Average LLM accuracy",
                    angle: -90,
                    position: "insideLeft",
                    offset: -2,
                    fill: "#94a3b8",
                  }}
                />
                <ReferenceLine
                  segment={[
                    { x: 0, y: 0 },
                    { x: 100, y: 100 },
                  ]}
                  stroke="rgba(226, 232, 240, 0.38)"
                  strokeDasharray="6 6"
                />
                <Tooltip
                  cursor={{ strokeDasharray: "4 4" }}
                  contentStyle={tooltipStyle()}
                  formatter={(value, name) => [
                    `${numericValue(value).toFixed(1)}%`,
                    name === "agreement_pct" ? "Average agreement" : "Average LLM accuracy",
                  ]}
                  labelFormatter={(_, payload) =>
                    payload?.[0] && "payload" in payload[0] && payload[0].payload
                      ? String((payload[0].payload as { model_name: string }).model_name)
                      : ""
                  }
                />
                <Scatter data={accuracyAgreementScatterData} shape={<ScatterPoint />}>
                  <LabelList dataKey="model_name" position="top" fill="#e2e8f0" fontSize={11} />
                  {accuracyAgreementScatterData.map((entry) => (
                    <Cell key={entry.model_name} fill={entry.fill} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {averageAccuracyGap != null ? (
          <p className="table-note">
            Across accuracy-aware models, agreement and LLM accuracy differ by{" "}
            {Math.round(averageAccuracyGap)} points on average.
          </p>
        ) : null}
      </section>

      <section className="visual-card dashboard-section-card">
        <SectionHeader
          eyebrow="Agreement Comparison"
          title="Where does consensus pull away from correctness?"
          description="Blue bars show average agreement. Green bars show LLM accuracy, so gaps reveal when models agree more than they are actually right."
        />

        {agreementComparisonData.length === 0 ||
        agreementComparisonData.every((row) => row.llm_accuracy_pct == null) ? (
          <p className="empty-state">
            No accuracy data is available for the loaded models, so the dashboard cannot overlay
            correctness on top of agreement yet.
          </p>
        ) : (
          <div className="chart-stage-large">
            <ResponsiveContainer
              width="100%"
              height={Math.max(240, agreementComparisonData.length * 64)}
            >
              <BarChart
                data={agreementComparisonData}
                layout="vertical"
                margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
                barGap={4}
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
                  formatter={(value, name) => [
                    `${numericValue(value).toFixed(1)}%`,
                    name === "agreement_pct" ? "Average agreement" : "LLM accuracy",
                  ]}
                />
                <Bar
                  dataKey="agreement_pct"
                  name="agreement_pct"
                  fill="#4c9aff"
                  fillOpacity={0.92}
                  radius={[0, 12, 12, 0]}
                />
                <Bar
                  dataKey="llm_accuracy_pct"
                  name="llm_accuracy_pct"
                  fill="#22c55e"
                  fillOpacity={0.58}
                  radius={[0, 12, 12, 0]}
                >
                  <LabelList
                    dataKey="llm_accuracy_pct"
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

      {costScatterData.length > 1 ? (
        <section className="visual-card dashboard-section-card">
          <SectionHeader
            eyebrow="Cost Efficiency"
            title="Who delivers the best accuracy per dollar?"
            description="Each dot is a model. Up is better verified accuracy, left is lower total cost."
          />

          <div className="chart-stage-large">
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 12, right: 36, left: 12, bottom: 18 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                <XAxis
                  type="number"
                  dataKey="total_cost"
                  domain={costScatterXDomain ?? [0, 1]}
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  tickFormatter={(value) => `$${numericValue(value).toFixed(2)}`}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="number"
                  dataKey="primary_accuracy_pct"
                  domain={costScatterYDomain ?? [0, 100]}
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
                    name === "total_cost" ? "Total cost" : "Best available accuracy",
                  ]}
                  labelFormatter={(_, payload) =>
                    payload?.[0] && "payload" in payload[0] && payload[0].payload
                      ? String((payload[0].payload as { model_name: string }).model_name)
                      : ""
                  }
                />
                <Scatter data={costScatterData} shape={<ScatterPoint />}>
                  <LabelList dataKey="model_name" position="top" fill="#e2e8f0" fontSize={11} />
                  {costScatterData.map((entry) => (
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
