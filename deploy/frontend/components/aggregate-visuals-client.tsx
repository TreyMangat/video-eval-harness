"use client";

import dynamic from "next/dynamic";

import type { AggregateModelRow } from "./aggregate-dashboard";

const AggregateVisuals = dynamic(
  () => import("./aggregate-visuals").then((module) => module.AggregateVisuals),
  { ssr: false }
);

export function AggregateVisualsClient({ rows }: { rows: AggregateModelRow[] }) {
  return <AggregateVisuals rows={rows} />;
}
