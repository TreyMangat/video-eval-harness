import Link from "next/link";

import {
  bestOverallModel,
  bestValueModel,
  buildCoreComparisonRows,
  displayRunName,
  formatMoney,
  formatPercent,
  getSweepData,
} from "../lib/analysis";
import type { RunPayload } from "../lib/types";

function shortModelName(name: string): string {
  return name
    .replace(/^(openai|google|anthropic|meta|qwen|alibaba|mistralai|x-ai)\//, "")
    .replace(/^(nvidia|novartis|deepseek)\//, "");
}

export function HomepageHero({ run }: { run: RunPayload }) {
  const sweepData = getSweepData(run);
  const rows = buildCoreComparisonRows(run, sweepData);
  const segmentCount = run.segments?.length ?? 0;

  const winner = bestOverallModel(rows);
  const value = bestValueModel(rows, segmentCount);

  const reportName =
    run.config.display_name?.trim() || displayRunName(run.run_id, run.config.created_at);
  const modelCount = Object.keys(run.summaries ?? {}).length;
  const videoCount = run.config.video_ids?.length ?? 0;
  const totalCost = Object.values(run.summaries ?? {}).reduce(
    (sum, s) => sum + (s.total_estimated_cost ?? 0),
    0
  );

  const winnerAccuracy = winner?.accuracy ?? winner?.llm_accuracy ?? winner?.agreement;
  const valueAccuracy = value?.accuracy ?? value?.llm_accuracy ?? value?.agreement;

  const modelNames = Object.keys(run.summaries ?? {}).map(shortModelName);
  const namedModels = modelNames.slice(0, 4);
  const remaining = modelNames.length - namedModels.length;

  return (
    <section className="homepage-hero">
      <p className="homepage-hero-eyebrow">VBench</p>

      <h1 className="homepage-hero-headline">
        How {modelCount} frontier AI models interpret the same video content.
      </h1>

      <p className="homepage-hero-desc">
        I benchmarked {namedModels.join(", ")}
        {remaining > 0 ? `, and ${remaining} others` : ""} on {videoCount} videos from UCF-101,
        measuring accuracy, agreement, cost, and latency across {segmentCount} segments.
      </p>

      <div className="homepage-hero-featured">
        <p className="homepage-hero-featured-label">Featured: {reportName}</p>

        <p className="homepage-hero-featured-meta">
          {modelCount} models &middot; {videoCount} videos &middot; {segmentCount} segments
          {totalCost != null ? <> &middot; ${totalCost.toFixed(2)}</> : null}
        </p>

        <div className="homepage-hero-cards">
          {winner && (
            <div className="homepage-hero-card winner">
              <span className="homepage-hero-card-label">Winner</span>
              <span className="homepage-hero-card-model">{shortModelName(winner.model_name)}</span>
              <span className="homepage-hero-card-stat">
                {winnerAccuracy != null ? formatPercent(winnerAccuracy) : "-"}
                {" "}&middot;{" "}
                {formatMoney(winner.total_cost)}
              </span>
            </div>
          )}
          {value && value.model_name !== winner?.model_name && (
            <div className="homepage-hero-card value">
              <span className="homepage-hero-card-label">Best Value</span>
              <span className="homepage-hero-card-model">{shortModelName(value.model_name)}</span>
              <span className="homepage-hero-card-stat">
                {valueAccuracy != null ? formatPercent(valueAccuracy) : "-"}
                {" "}&middot;{" "}
                {formatMoney(value.total_cost)}
              </span>
            </div>
          )}
        </div>

        <div className="homepage-hero-actions">
          <Link href={`/report/${run.run_id}`} className="homepage-hero-cta primary">
            View full report &rarr;
          </Link>
          <Link href="/runs" className="homepage-hero-cta secondary">
            Browse all runs &rarr;
          </Link>
        </div>
      </div>
    </section>
  );
}
