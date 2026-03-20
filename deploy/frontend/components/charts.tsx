"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ModelSummary } from "../lib/types";

const MODEL_COLORS = [
  "#f59e0b",
  "#22c55e",
  "#3b82f6",
  "#a855f7",
  "#ef4444",
  "#06b6d4",
  "#ec4899",
  "#84cc16",
];

function getColor(index: number): string {
  return MODEL_COLORS[index % MODEL_COLORS.length];
}

function coerceNumber(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    return Number(value);
  }
  if (Array.isArray(value) && value.length > 0) {
    return coerceNumber(value[0]);
  }
  return 0;
}

type ChartProps = {
  summaries: Record<string, ModelSummary>;
  models: string[];
};

export function LatencyChart({ summaries, models }: ChartProps) {
  const data = models.map((m, i) => ({
    name: m,
    avg: Math.round(summaries[m]?.avg_latency_ms ?? 0),
    p95: Math.round(summaries[m]?.p95_latency_ms ?? 0),
    fill: getColor(i),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} barGap={4} barCategoryGap="20%">
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(20,40,29,0.08)" />
        <XAxis
          dataKey="name"
          tick={{ fontSize: 12 }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          tickLine={false}
          axisLine={false}
          label={{ value: "ms", angle: -90, position: "insideLeft", fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{
            background: "rgba(255,252,245,0.95)",
            border: "1px solid rgba(20,40,29,0.12)",
            borderRadius: 12,
            fontSize: 13,
          }}
          formatter={(value, name) => [
            `${coerceNumber(value)} ms`,
            String(name) === "avg" ? "Avg Latency" : "P95 Latency",
          ]}
        />
        <Bar dataKey="avg" name="avg" radius={[6, 6, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.fill} fillOpacity={0.85} />
          ))}
        </Bar>
        <Bar dataKey="p95" name="p95" radius={[6, 6, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.fill} fillOpacity={0.45} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CostChart({ summaries, models }: ChartProps) {
  const data = models.map((m, i) => ({
    name: m,
    cost: Number((summaries[m]?.total_estimated_cost ?? 0).toFixed(4)),
    fill: getColor(i),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} barCategoryGap="30%">
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(20,40,29,0.08)" />
        <XAxis
          dataKey="name"
          tick={{ fontSize: 12 }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          tickLine={false}
          axisLine={false}
          label={{ value: "USD", angle: -90, position: "insideLeft", fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{
            background: "rgba(255,252,245,0.95)",
            border: "1px solid rgba(20,40,29,0.12)",
            borderRadius: 12,
            fontSize: 13,
          }}
          formatter={(value) => [`$${coerceNumber(value).toFixed(4)}`, "Total Cost"]}
        />
        <Bar dataKey="cost" radius={[6, 6, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.fill} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ConfidenceChart({ summaries, models }: ChartProps) {
  const data = models.map((m, i) => ({
    name: m,
    confidence: Number((summaries[m]?.avg_confidence ?? 0).toFixed(3)),
    fill: getColor(i),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} barCategoryGap="30%">
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(20,40,29,0.08)" />
        <XAxis
          dataKey="name"
          tick={{ fontSize: 12 }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fontSize: 12 }}
          tickLine={false}
          axisLine={false}
          label={{ value: "Score", angle: -90, position: "insideLeft", fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{
            background: "rgba(255,252,245,0.95)",
            border: "1px solid rgba(20,40,29,0.12)",
            borderRadius: 12,
            fontSize: 13,
          }}
          formatter={(value) => [coerceNumber(value).toFixed(3), "Avg Confidence"]}
        />
        <Bar dataKey="confidence" radius={[6, 6, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.fill} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ParseRateChart({ summaries, models }: ChartProps) {
  const data = models.map((m, i) => ({
    name: m,
    rate: Number(((summaries[m]?.parse_success_rate ?? 0) * 100).toFixed(1)),
    fill: getColor(i),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} barCategoryGap="30%">
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(20,40,29,0.08)" />
        <XAxis
          dataKey="name"
          tick={{ fontSize: 12 }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 12 }}
          tickLine={false}
          axisLine={false}
          label={{ value: "%", angle: -90, position: "insideLeft", fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{
            background: "rgba(255,252,245,0.95)",
            border: "1px solid rgba(20,40,29,0.12)",
            borderRadius: 12,
            fontSize: 13,
          }}
          formatter={(value) => [`${coerceNumber(value)}%`, "Parse Success"]}
        />
        <Bar dataKey="rate" radius={[6, 6, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.fill} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
