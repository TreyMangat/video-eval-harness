"use client";

import Link from "next/link";
import { useDeferredValue, useMemo, useState } from "react";

import { formatDateTime } from "../lib/analysis";
import { getRunType, isAccuracyTestRun, isBenchmarkRun, isComparisonRun } from "../lib/run-type";
import { RunTypeBadge } from "./run-type-badge";

export type RunsTableRow = {
  run_id: string;
  display_name: string;
  created_at: string;
  models: string[];
  video_names: string[];
  best_agreement: number | null;
  best_model_name: string | null;
  run_type?: "comparison" | "accuracy_test" | "benchmark" | null;
  has_accuracy?: boolean;
  data_dir?: string;
};

function formatPercent(value: number | null): string {
  if (value == null) {
    return "\u2014";
  }
  return `${Math.round(value * 100)}%`;
}

function buildHref(pathname: string, query: Record<string, string | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value) {
      params.set(key, value);
    }
  }
  const suffix = params.toString();
  return suffix ? `${pathname}?${suffix}` : pathname;
}

export function RunsTable({ rows }: { rows: RunsTableRow[] }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | "accuracy" | "comparison" | "benchmark">("all");
  const deferredQuery = useDeferredValue(query);

  const counts = useMemo(
    () => ({
      all: rows.length,
      accuracy: rows.filter((row) => isAccuracyTestRun(row)).length,
      comparison: rows.filter((row) => isComparisonRun(row)).length,
      benchmark: rows.filter((row) => isBenchmarkRun(row)).length,
    }),
    [rows]
  );

  const filteredRows = useMemo(() => {
    const typeFilteredRows = rows.filter((row) => {
      const runType = getRunType(row);
      if (filter === "accuracy") {
        return runType === "accuracy_test";
      }
      if (filter === "comparison") {
        return runType === "comparison";
      }
      if (filter === "benchmark") {
        return runType === "benchmark";
      }
      return true;
    });

    const normalized = deferredQuery.trim().toLowerCase();
    if (!normalized) {
      return typeFilteredRows;
    }

    return typeFilteredRows.filter((row) => {
      const haystack = [
        row.display_name,
        row.run_id,
        row.models.join(" "),
        row.video_names.join(" "),
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(normalized);
    });
  }, [deferredQuery, filter, rows]);

  return (
    <section className="visual-card">
      <div className="section-heading">
        <p className="section-eyebrow">Run Index</p>
        <h2 className="run-title">Which run do you want to inspect?</h2>
        <p className="chart-desc">
          Filter by run name, model, or video, then jump straight into the dashboard, report, or
          segment story.
        </p>
      </div>

      <div className="run-filter-tabs" aria-label="Filter runs by type">
        <button
          type="button"
          className={`filter-tab ${filter === "all" ? "active" : ""}`}
          onClick={() => setFilter("all")}
        >
          All runs ({counts.all})
        </button>
        <button
          type="button"
          className={`filter-tab ${filter === "accuracy" ? "active" : ""}`}
          onClick={() => setFilter("accuracy")}
        >
          Accuracy tests ({counts.accuracy})
        </button>
        <button
          type="button"
          className={`filter-tab ${filter === "comparison" ? "active" : ""}`}
          onClick={() => setFilter("comparison")}
        >
          Comparisons ({counts.comparison})
        </button>
        <button
          type="button"
          className={`filter-tab ${filter === "benchmark" ? "active" : ""}`}
          onClick={() => setFilter("benchmark")}
        >
          Benchmarks ({counts.benchmark})
        </button>
      </div>

      <div className="runs-filter-row">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search runs, models, or videos"
          aria-label="Filter runs"
        />
      </div>

      {filteredRows.length === 0 ? (
        <p className="empty-state">No runs match that filter.</p>
      ) : (
        <div className="table-scroll">
          <table className="data-table runs-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Date</th>
                <th>Models</th>
                <th>Videos</th>
                <th>Quick Stats</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row) => (
                <tr key={row.run_id}>
                  <td data-label="Name">
                    <div className="run-list-cell">
                      <div className="run-list-title-row">
                        <p className="run-list-name">{row.display_name}</p>
                        <RunTypeBadge run={row} />
                      </div>
                    </div>
                  </td>
                  <td data-label="Date">{formatDateTime(row.created_at)}</td>
                  <td data-label="Models" title={row.models.join(", ")}>
                    {row.models.length} {row.models.length === 1 ? "model" : "models"}
                  </td>
                  <td data-label="Videos" title={row.video_names.join(", ")}>
                    {row.video_names.length} {row.video_names.length === 1 ? "video" : "videos"}
                  </td>
                  <td data-label="Quick Stats">
                    <div className="quick-stat-cell">
                      <strong>{formatPercent(row.best_agreement)}</strong>
                      <span>{row.best_model_name ?? "Not enough data"}</span>
                    </div>
                  </td>
                  <td data-label="Actions">
                    <div className="inline-links">
                      <Link
                        href={buildHref(`/report/${row.run_id}`, {
                          dataDir: row.data_dir,
                        })}
                      >
                        Report
                      </Link>
                      <Link
                        href={buildHref(`/runs/${row.run_id}/segments`, {
                          dataDir: row.data_dir,
                        })}
                      >
                        Segments
                      </Link>
                      <Link
                        href={buildHref(`/runs/${row.run_id}/cost`, {
                          dataDir: row.data_dir,
                        })}
                      >
                        Cost
                      </Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
