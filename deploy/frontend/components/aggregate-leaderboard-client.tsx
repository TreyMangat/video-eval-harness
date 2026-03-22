"use client";

import { useMemo, useState } from "react";

import {
  formatLatency,
  formatMoney,
  formatPercent,
  getModelTier,
  modelColor,
} from "../lib/analysis";
import type { AggregateModelRow } from "./aggregate-dashboard";

type SortKey =
  | "model_name"
  | "avg_agreement"
  | "avg_accuracy"
  | "avg_llm_accuracy"
  | "avg_confidence"
  | "avg_latency_ms"
  | "total_cost"
  | "run_count";

type SortDirection = "asc" | "desc";

function preferredAccuracy(row: AggregateModelRow): number | null {
  return row.avg_llm_accuracy ?? row.avg_accuracy;
}

function signalIndicator(row: AggregateModelRow): { label: string; className: string } | null {
  const accuracy = preferredAccuracy(row);
  if (accuracy == null || row.avg_agreement == null) {
    return null;
  }

  const delta = row.avg_agreement - accuracy;
  if (delta >= 0.1) {
    return {
      label: `overconfident +${Math.round(delta * 100)} pts`,
      className: "aggregate-signal-badge warning",
    };
  }
  if (delta <= -0.1) {
    return {
      label: `underrated +${Math.round(Math.abs(delta) * 100)} pts`,
      className: "aggregate-signal-badge positive",
    };
  }
  return null;
}

function formatConfidence(value: number | null): string {
  if (value == null) {
    return "-";
  }
  return value.toFixed(2);
}

function confidenceClass(value: number | null): string {
  if (value == null) {
    return "";
  }
  if (value >= 0.8) {
    return "confidence-high";
  }
  if (value >= 0.5) {
    return "confidence-mid";
  }
  return "confidence-low";
}

function accuracyClass(value: number | null): string {
  if (value == null) {
    return "aggregate-accuracy-chip neutral";
  }
  if (value >= 0.8) {
    return "aggregate-accuracy-chip high";
  }
  if (value >= 0.6) {
    return "aggregate-accuracy-chip mid";
  }
  return "aggregate-accuracy-chip low";
}

function scoreTone(value: number | null): "high" | "mid" | "low" | "none" {
  if (value == null) {
    return "none";
  }
  if (value >= 0.8) {
    return "high";
  }
  if (value >= 0.5) {
    return "mid";
  }
  return "low";
}

function medalClass(index: number): string {
  if (index === 0) {
    return "medal-gold";
  }
  if (index === 1) {
    return "medal-silver";
  }
  if (index === 2) {
    return "medal-bronze";
  }
  return "";
}

function tierBadgeLabel(modelName: string): string {
  const tier = getModelTier(modelName);
  if (tier === "free") {
    return "Free";
  }
  if (tier === "fast") {
    return "Fast";
  }
  if (tier === "frontier") {
    return "Frontier";
  }
  return "Unknown";
}

function defaultSortKey(rows: AggregateModelRow[]): SortKey {
  if (rows.some((row) => row.avg_llm_accuracy != null)) {
    return "avg_llm_accuracy";
  }
  if (rows.some((row) => row.avg_accuracy != null)) {
    return "avg_accuracy";
  }
  return "avg_agreement";
}

function defaultDirectionFor(key: SortKey): SortDirection {
  if (key === "model_name") {
    return "asc";
  }
  if (key === "avg_latency_ms" || key === "total_cost") {
    return "asc";
  }
  return "desc";
}

function compareNullableNumber(
  left: number | null,
  right: number | null,
  direction: SortDirection
): number {
  if (left == null && right == null) {
    return 0;
  }
  if (left == null) {
    return 1;
  }
  if (right == null) {
    return -1;
  }
  return direction === "asc" ? left - right : right - left;
}

function compareRows(
  left: AggregateModelRow,
  right: AggregateModelRow,
  sortKey: SortKey,
  sortDirection: SortDirection
): number {
  if (sortKey === "model_name") {
    return sortDirection === "asc"
      ? left.model_name.localeCompare(right.model_name)
      : right.model_name.localeCompare(left.model_name);
  }

  const leftValue =
    sortKey === "avg_agreement"
      ? left.avg_agreement
      : sortKey === "avg_accuracy"
        ? left.avg_accuracy
        : sortKey === "avg_llm_accuracy"
          ? left.avg_llm_accuracy
          : sortKey === "avg_confidence"
            ? left.avg_confidence
            : sortKey === "avg_latency_ms"
              ? left.avg_latency_ms
              : sortKey === "total_cost"
                ? left.total_cost
                : sortKey === "run_count"
                  ? left.run_count
                  : null;

  const rightValue =
    sortKey === "avg_agreement"
      ? right.avg_agreement
      : sortKey === "avg_accuracy"
        ? right.avg_accuracy
        : sortKey === "avg_llm_accuracy"
          ? right.avg_llm_accuracy
          : sortKey === "avg_confidence"
            ? right.avg_confidence
            : sortKey === "avg_latency_ms"
              ? right.avg_latency_ms
              : sortKey === "total_cost"
                ? right.total_cost
                : sortKey === "run_count"
                  ? right.run_count
                  : null;

  const primaryCompare = compareNullableNumber(leftValue, rightValue, sortDirection);
  if (primaryCompare !== 0) {
    return primaryCompare;
  }

  const accuracyCompare = compareNullableNumber(
    preferredAccuracy(left),
    preferredAccuracy(right),
    "desc"
  );
  if (accuracyCompare !== 0) {
    return accuracyCompare;
  }

  const agreementCompare = compareNullableNumber(left.avg_agreement, right.avg_agreement, "desc");
  if (agreementCompare !== 0) {
    return agreementCompare;
  }

  const runCompare = compareNullableNumber(left.run_count, right.run_count, "desc");
  if (runCompare !== 0) {
    return runCompare;
  }

  return left.model_name.localeCompare(right.model_name);
}

function SortableHeader({
  label,
  sortKey,
  activeKey,
  direction,
  onSort,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  direction: SortDirection;
  onSort: (key: SortKey) => void;
}) {
  return (
    <button
      type="button"
      className={`leaderboard-sort ${activeKey === sortKey ? "is-active" : ""}`}
      onClick={() => onSort(sortKey)}
    >
      <span>{label}</span>
      <span className="leaderboard-sort-arrow">
        {activeKey === sortKey ? (direction === "asc" ? "\u25b2" : "\u25bc") : ""}
      </span>
    </button>
  );
}

export function AggregateLeaderboardClient({
  rows,
  hasExactAccuracy,
  hasLlmAccuracy,
}: {
  rows: AggregateModelRow[];
  hasExactAccuracy: boolean;
  hasLlmAccuracy: boolean;
}) {
  const initialSortKey = defaultSortKey(rows);
  const [sortKey, setSortKey] = useState<SortKey>(initialSortKey);
  const [sortDirection, setSortDirection] = useState<SortDirection>(
    defaultDirectionFor(initialSortKey)
  );

  const sortedRows = useMemo(
    () => [...rows].sort((left, right) => compareRows(left, right, sortKey, sortDirection)),
    [rows, sortDirection, sortKey]
  );

  function handleSort(nextSortKey: SortKey) {
    if (nextSortKey === sortKey) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }

    setSortKey(nextSortKey);
    setSortDirection(defaultDirectionFor(nextSortKey));
  }

  return (
    <div className="leaderboard-scroll">
      <table className="leaderboard-table aggregate-leaderboard-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>
              <SortableHeader
                label="Model"
                sortKey="model_name"
                activeKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
              />
            </th>
            <th>
              <SortableHeader
                label="Avg Agreement"
                sortKey="avg_agreement"
                activeKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
              />
            </th>
            {hasExactAccuracy ? (
              <th>
                <SortableHeader
                  label="Avg Accuracy"
                  sortKey="avg_accuracy"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                />
              </th>
            ) : null}
            {hasLlmAccuracy ? (
              <th>
                <SortableHeader
                  label="LLM Accuracy"
                  sortKey="avg_llm_accuracy"
                  activeKey={sortKey}
                  direction={sortDirection}
                  onSort={handleSort}
                />
              </th>
            ) : null}
            <th>
              <SortableHeader
                label="Avg Confidence"
                sortKey="avg_confidence"
                activeKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
              />
            </th>
            <th>
              <SortableHeader
                label="Avg Latency"
                sortKey="avg_latency_ms"
                activeKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
              />
            </th>
            <th>
              <SortableHeader
                label="Total Cost"
                sortKey="total_cost"
                activeKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
              />
            </th>
            <th>
              <SortableHeader
                label="Runs"
                sortKey="run_count"
                activeKey={sortKey}
                direction={sortDirection}
                onSort={handleSort}
              />
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row, index) => {
            const medal = medalClass(index);
            const agreementTone = scoreTone(row.avg_agreement);
            const signal = signalIndicator(row);
            return (
              <tr
                key={row.model_name}
                className={`leaderboard-row aggregate-leaderboard-row ${index === 0 ? "top-ranked" : ""} ${medal}`}
                style={{ boxShadow: `inset 4px 0 0 ${modelColor(row.model_name)}` }}
              >
                <td className={`leaderboard-rank ${medal}`}>
                  <span className={`leaderboard-rank-pill ${medal}`}>#{index + 1}</span>
                </td>
                <td>
                  <div className="aggregate-model-cell">
                    <div className="aggregate-model-heading">
                      <strong>{row.model_name}</strong>
                      <span className={`aggregate-tier-badge ${getModelTier(row.model_name)}`}>
                        {tierBadgeLabel(row.model_name)}
                      </span>
                    </div>
                    <span>
                      {row.run_count} {row.run_count === 1 ? "run" : "runs"} loaded
                    </span>
                    {signal ? <span className={signal.className}>{signal.label}</span> : null}
                  </div>
                </td>
                <td>
                  <div className={`aggregate-score-track ${agreementTone}`}>
                    <span
                      className={`aggregate-score-fill ${agreementTone}`}
                      style={{
                        width: `${Math.max(0, Math.min(100, (row.avg_agreement ?? 0) * 100))}%`,
                      }}
                    />
                    <span className="aggregate-score-label">{formatPercent(row.avg_agreement)}</span>
                  </div>
                </td>
                {hasExactAccuracy ? (
                  <td>
                    <span className={accuracyClass(row.avg_accuracy)}>
                      {formatPercent(row.avg_accuracy)}
                    </span>
                  </td>
                ) : null}
                {hasLlmAccuracy ? (
                  <td>
                    <span className={accuracyClass(row.avg_llm_accuracy)}>
                      {formatPercent(row.avg_llm_accuracy)}
                    </span>
                  </td>
                ) : null}
                <td className={confidenceClass(row.avg_confidence)}>
                  {formatConfidence(row.avg_confidence)}
                </td>
                <td>{formatLatency(row.avg_latency_ms)}</td>
                <td>{formatMoney(row.total_cost)}</td>
                <td>{row.run_count}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
