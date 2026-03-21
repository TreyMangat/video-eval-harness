"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

import { displayRunName } from "../lib/analysis";
import type { RunListItem } from "../lib/types";

type RunSelectorConfig = {
  runs: RunListItem[];
  selectedRunId: string;
  dataDir?: string;
  basePath?: string;
};

export function TopNav({
  active,
  runSelector,
}: {
  active: "dashboard" | "runs" | "compare";
  runSelector?: RunSelectorConfig;
}) {
  const router = useRouter();

  function handleRunChange(runId: string): void {
    if (!runSelector) {
      return;
    }

    const params = new URLSearchParams();
    params.set("run", runId);
    if (runSelector.dataDir) {
      params.set("dataDir", runSelector.dataDir);
    }
    const basePath = runSelector.basePath ?? "/";
    router.push(`${basePath}?${params.toString()}`);
  }

  return (
    <header className="top-bar">
      <Link
        href="/"
        className="top-bar-left"
        style={{ textDecoration: "none", color: "inherit" }}
      >
        <h1 className="brand">VBench</h1>
        <span className="brand-sub">Benchmark results, organized for decisions</span>
      </Link>
      <div className="top-bar-right">
        <nav className="mode-toggle" aria-label="Primary">
          <Link href="/" className={`mode-btn ${active === "dashboard" ? "active" : ""}`}>
            Dashboard
          </Link>
          <Link href="/runs" className={`mode-btn ${active === "runs" ? "active" : ""}`}>
            Runs
          </Link>
          <Link href="/compare" className={`mode-btn ${active === "compare" ? "active" : ""}`}>
            Compare
          </Link>
        </nav>
        {runSelector ? (
          <label className="top-run-select">
            <span>Run</span>
            <select
              aria-label="Select run"
              value={runSelector.selectedRunId}
              onChange={(event) => handleRunChange(event.target.value)}
            >
              {runSelector.runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {displayRunName(run.run_id, run.created_at)}
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>
    </header>
  );
}
