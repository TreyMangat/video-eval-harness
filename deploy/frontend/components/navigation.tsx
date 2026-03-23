"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { displayRunName } from "../lib/analysis";
import type { RunListItem } from "../lib/types";

type RunSelectorConfig = {
  runs: RunListItem[];
  selectedRunId: string;
  dataDir?: string;
  basePath?: string;
};

const PRIMARY_NAV_ITEMS = [
  { href: "/", label: "Dashboard", key: "dashboard" },
  { href: "/runs", label: "Runs", key: "runs" },
  { href: "/compare", label: "Compare", key: "compare" },
  { href: "/accuracy-test", label: "Accuracy Test", key: "accuracy" },
  { href: "/batch-accuracy-test", label: "Batch Test", key: "batch" },
  { href: "/new", label: "New Benchmark", key: "new" },
] as const;

export function TopNav({
  active,
  runSelector,
}: {
  active: "dashboard" | "runs" | "compare" | "accuracy" | "batch" | "new";
  runSelector?: RunSelectorConfig;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

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

  function renderPrimaryLinks(
    className = "mode-btn",
    onNavigate?: () => void
  ) {
    return PRIMARY_NAV_ITEMS.map((item) => (
      <Link
        key={item.href}
        href={item.href}
        className={`${className} ${active === item.key ? "active" : ""}`}
        onClick={onNavigate}
      >
        {item.label}
      </Link>
    ));
  }

  function renderRunSelector(className = "top-run-select") {
    if (!runSelector) {
      return null;
    }

    return (
      <label className={className}>
        <span>Run</span>
        <select
          className="run-selector"
          aria-label="Select run"
          value={runSelector.selectedRunId}
          onChange={(event) => {
            handleRunChange(event.target.value);
            setMobileMenuOpen(false);
          }}
        >
          {runSelector.runs.map((run) => (
            <option key={run.run_id} value={run.run_id}>
              {displayRunName(run.run_id, run.created_at)}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <header className={`top-bar ${mobileMenuOpen ? "mobile-open" : ""}`}>
      <Link
        href="/"
        className="top-bar-left"
        style={{ textDecoration: "none", color: "inherit" }}
      >
        <h1 className="brand">VBench</h1>
        <span className="brand-sub">Benchmark results, organized for decisions</span>
      </Link>
      <div className="top-bar-right">
        <nav className="mode-toggle desktop-nav" aria-label="Primary">
          {renderPrimaryLinks()}
        </nav>
        {renderRunSelector("top-run-select desktop-run-select")}
        <button
          type="button"
          className="mobile-menu-btn"
          aria-expanded={mobileMenuOpen}
          aria-controls="mobile-primary-nav"
          aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
          onClick={() => setMobileMenuOpen((current) => !current)}
        >
          <span className="mobile-menu-line" />
          <span className="mobile-menu-line" />
          <span className="mobile-menu-line" />
        </button>
      </div>
      {mobileMenuOpen ? (
        <div id="mobile-primary-nav" className="mobile-menu">
          <nav className="mobile-nav-links" aria-label="Mobile primary">
            {renderPrimaryLinks("mobile-nav-link", () => setMobileMenuOpen(false))}
          </nav>
          {renderRunSelector("top-run-select mobile-run-select")}
        </div>
      ) : null}
    </header>
  );
}
