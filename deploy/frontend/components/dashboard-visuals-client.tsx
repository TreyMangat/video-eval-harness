"use client";

import dynamic from "next/dynamic";

import type { CoreComparisonRow, CostBreakdown } from "../lib/analysis";
import type { RunPayload, SweepMetrics } from "../lib/types";

const DashboardVisuals = dynamic(
  () => import("./dashboard-visuals").then((module) => module.DashboardVisuals),
  { ssr: false }
);

export function DashboardVisualsClient({
  run,
  rows,
  costBreakdown,
  sweepData,
}: {
  run: RunPayload;
  rows: CoreComparisonRow[];
  costBreakdown: CostBreakdown | null;
  sweepData: SweepMetrics | null;
}) {
  return (
    <DashboardVisuals
      run={run}
      rows={rows}
      costBreakdown={costBreakdown}
      sweepData={sweepData}
    />
  );
}
