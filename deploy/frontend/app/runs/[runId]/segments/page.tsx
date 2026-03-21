import Link from "next/link";
import { notFound } from "next/navigation";

import {
  displayRunName,
  displaySegmentName,
  displayVideoName,
  formatLatency,
  formatMoney,
  formatPercent,
  formatTime,
  getSweepData,
  groupSegmentsByVideo,
  modelColor,
  resultVariantLabel,
} from "../../../../lib/analysis";
import { loadRun, loadSegmentMedia } from "../../../../lib/run-source";
import { RunMetadataCard } from "../../../../components/analysis-panels";
import { TopNav } from "../../../../components/navigation";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ALL_VARIANTS = "All variants";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function buildQuery(
  runId: string,
  options: {
    dataDir?: string;
    video?: string;
    variant?: string;
  }
): string {
  const params = new URLSearchParams();
  if (options.dataDir) {
    params.set("dataDir", options.dataDir);
  }
  if (options.video) {
    params.set("video", options.video);
  }
  if (options.variant) {
    params.set("variant", options.variant);
  }
  const query = params.toString();
  return query ? `/runs/${runId}/segments?${query}` : `/runs/${runId}/segments`;
}

function segmentHasDisagreement(
  results: Array<{ primary_action: string | null }>
): boolean {
  const actions = [...new Set(results.map((result) => result.primary_action?.trim()).filter(Boolean))];
  return actions.length > 1;
}

export default async function RunSegmentsPage({
  params,
  searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{
    dataDir?: string | string[];
    video?: string | string[];
    variant?: string | string[];
  }>;
}) {
  const { runId } = await params;
  const resolvedSearchParams = await searchParams;
  const dataDir = readFirst(resolvedSearchParams.dataDir) ?? process.env.VBENCH_RUNS_DIR;
  const run = await loadRun(runId, dataDir);

  if (!run) {
    notFound();
  }

  const sweepData = getSweepData(run);
  const segmentsByVideo = groupSegmentsByVideo(run);
  const videoIds = Object.keys(segmentsByVideo).sort();
  const selectedVideoId = readFirst(resolvedSearchParams.video) ?? videoIds[0];
  const selectedVideoSegments = segmentsByVideo[selectedVideoId] ?? [];
  const defaultVariant = sweepData?.variants[0] ?? ALL_VARIANTS;
  const selectedVariant = readFirst(resolvedSearchParams.variant) ?? defaultVariant;
  const selectedVariantId =
    sweepData && selectedVariant !== ALL_VARIANTS
      ? sweepData.variant_id_by_label[selectedVariant] ?? null
      : null;
  const modelOrder = new Map(run.models.map((modelName, index) => [modelName, index]));
  const segmentStories = await Promise.all(
    selectedVideoSegments.map(async (segment) => {
      const media = await loadSegmentMedia(
        run.run_id,
        segment.segment_id,
        selectedVariantId,
        dataDir
      );
      const results = run.results
        .filter((result) => result.segment_id === segment.segment_id)
        .filter((result) =>
          selectedVariant === ALL_VARIANTS ? true : resultVariantLabel(result) === selectedVariant
        )
        .sort(
          (left, right) =>
            (modelOrder.get(left.model_name) ?? Number.MAX_SAFE_INTEGER) -
              (modelOrder.get(right.model_name) ?? Number.MAX_SAFE_INTEGER) ||
            resultVariantLabel(left).localeCompare(resultVariantLabel(right))
        );

      return { segment, media, results };
    })
  );

  return (
    <main className="analysis-shell">
      <TopNav active="runs" />

      <RunMetadataCard
        title="Run Context"
        runId={run.run_id}
        createdAt={run.config.created_at}
        videoLabel=""
        promptVersion={run.config.prompt_version}
        models={run.models}
        segments={run.segments.length}
        compact
        compactText={[
          displayRunName(run.run_id, run.config.created_at),
          selectedVideoId ? displayVideoName(selectedVideoId) : "No video",
          selectedVariant,
        ].join(" \u00b7 ")}
      />

      <section className="visual-card run-detail-hero">
        <div className="dashboard-intro-top">
          <div>
            <p className="section-eyebrow">Segments</p>
            <h1 className="run-title">How does each segment look model by model?</h1>
            <p className="run-meta dashboard-copy">
              Select one video, then scan every segment with the extracted frames and model outputs
              side by side.
            </p>
          </div>
          <form method="get" className="run-select-form">
            <label className="field">
              <span>Video</span>
              <select name="video" defaultValue={selectedVideoId}>
                {videoIds.map((videoId) => (
                  <option key={videoId} value={videoId}>
                    {displayVideoName(videoId)}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Variant</span>
              <select name="variant" defaultValue={selectedVariant}>
                {sweepData ? (
                  <>
                    {sweepData.variants.map((variant) => (
                      <option key={variant} value={variant}>
                        {variant}
                      </option>
                    ))}
                    <option value={ALL_VARIANTS}>{ALL_VARIANTS}</option>
                  </>
                ) : (
                  <option value={ALL_VARIANTS}>{ALL_VARIANTS}</option>
                )}
              </select>
            </label>
            {dataDir ? <input type="hidden" name="dataDir" value={dataDir} /> : null}
            <button type="submit" className="primary-btn">
              Load Video
            </button>
          </form>
        </div>
      </section>

      <section className="visual-card">
        <div className="section-heading">
          <p className="section-eyebrow">Test Suite</p>
          <h3>Which video are you looking at?</h3>
          <p>Runs can contain multiple videos. Pick one and drill into its segment-by-segment story.</p>
        </div>
        <div className="video-breakdown-grid">
          {videoIds.map((videoId) => (
            <Link
              key={videoId}
              href={buildQuery(run.run_id, {
                dataDir,
                video: videoId,
                variant: selectedVariant,
              })}
              className={`video-breakdown-card ${videoId === selectedVideoId ? "active" : ""}`}
            >
              <p className="video-breakdown-label">Video</p>
              <h3>{displayVideoName(videoId)}</h3>
              <p>{segmentsByVideo[videoId]?.length ?? 0} segments</p>
            </Link>
          ))}
        </div>
      </section>

      {selectedVideoSegments.length ? (
        <section className="visual-card">
          <div className="section-heading">
            <p className="section-eyebrow">Jump To</p>
            <h3>Where should you start?</h3>
            <p>Use the segment chips for quick navigation, then scroll to inspect every answer card.</p>
          </div>
          <div className="segment-jump-list">
            {selectedVideoSegments.map((segment) => (
              <a key={segment.segment_id} href={`#${segment.segment_id}`} className="segment-jump-chip">
                <strong>{displayVideoName(segment.video_id)}</strong>
                <span>{formatTime(segment.start_time_s)} - {formatTime(segment.end_time_s)}</span>
              </a>
            ))}
          </div>
        </section>
      ) : null}

      {segmentStories.length ? (
        <div className="segment-story-stack">
          {segmentStories.map(({ segment, media, results }) => (
            <section key={segment.segment_id} id={segment.segment_id} className="segment-story-card">
              <div className="segment-story-head">
                <div>
                  <p className="section-eyebrow">Segment {segment.segment_index + 1}</p>
                  <h3>{displaySegmentName(segment)}</h3>
                  <p className="segment-story-meta">
                    {formatTime(segment.start_time_s)} - {formatTime(segment.end_time_s)} |{" "}
                    {Math.round(segment.duration_s)}s | {segment.frame_count} extracted frames
                  </p>
                </div>
                <div className="story-badge-stack">
                  {segmentHasDisagreement(results) ? (
                    <span className="warning-pill">Models disagree</span>
                  ) : null}
                  <span className="story-badge">
                    {results.length} model outputs
                    {selectedVariant !== ALL_VARIANTS ? ` \u00b7 ${selectedVariant}` : ""}
                  </span>
                </div>
              </div>

              {media?.frames.length ? (
                <div className="segment-thumb-row">
                  {media.frames.map((frame) => (
                    <figure
                      key={`${segment.segment_id}-${frame.timestamp_s}`}
                      className="segment-thumb"
                    >
                      {frame.data_url ? (
                        <img
                          src={frame.data_url}
                          alt={`Frame at ${frame.timestamp_s.toFixed(2)} seconds`}
                        />
                      ) : (
                        <div className="empty-state">Frame unavailable</div>
                      )}
                      <figcaption>{frame.timestamp_s.toFixed(2)}s</figcaption>
                    </figure>
                  ))}
                </div>
              ) : (
                <p className="empty-state">No extracted frame previews were exported for this segment.</p>
              )}

              <div className="segment-answer-grid">
                {results.map((result) => (
                  <article
                    key={`${segment.segment_id}-${result.model_name}-${resultVariantLabel(result)}`}
                    className="segment-answer-card"
                    style={{ borderTopColor: modelColor(result.model_name) }}
                  >
                    <div className="segment-answer-head">
                      <div>
                        <p className="segment-answer-model">{result.model_name}</p>
                        {selectedVariant === ALL_VARIANTS && sweepData ? (
                          <p className="segment-answer-variant">{resultVariantLabel(result)}</p>
                        ) : null}
                      </div>
                      <span className={`parse-badge ${result.parsed_success ? "ok" : "fail"}`}>
                        {result.parsed_success ? "Parsed" : "Failed"}
                      </span>
                    </div>

                    <p className="segment-answer-primary">{result.primary_action || "-"}</p>
                    <p className="segment-answer-secondary">
                      Confidence{" "}
                      {result.confidence == null ? "-" : result.confidence.toFixed(2)} |{" "}
                      {formatLatency(result.latency_ms)} | {formatMoney(result.estimated_cost)}
                    </p>
                    <p className="segment-answer-description">
                      {result.description || result.parse_error || "No description returned."}
                    </p>

                    {result.secondary_actions.length ? (
                      <div className="tag-list">
                        {result.secondary_actions.map((action) => (
                          <span key={`${result.model_name}-${action}`} className="action-tag">
                            {action}
                          </span>
                        ))}
                      </div>
                    ) : null}

                    {result.objects.length ? (
                      <div className="tag-list">
                        {result.objects.map((objectName) => (
                          <span key={`${result.model_name}-${objectName}`} className="object-tag">
                            {objectName}
                          </span>
                        ))}
                      </div>
                    ) : null}

                    <div className="segment-answer-footer">
                      <span>Parse {formatPercent(result.parsed_success ? 1 : 0)}</span>
                      <span>{result.provider}</span>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <section className="visual-card">
          <p className="empty-state">No segments are available for the selected video.</p>
        </section>
      )}
    </main>
  );
}
