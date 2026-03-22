import Link from "next/link";

import { TopNav } from "../../components/navigation";
import { UploadZone } from "../../components/upload-zone";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export default function NewBenchmarkPage() {
  return (
    <main className="analysis-shell">
      <TopNav active="new" />

      <section className="visual-card new-benchmark-hero dashboard-section-card">
        <div className="section-heading">
          <p className="section-eyebrow">New Benchmark</p>
          <h1 className="run-title">Upload a clip and launch a fresh run.</h1>
          <p className="chart-desc">
            Pick up to three fast-tier models, submit a short clip, and we&apos;ll route you to the
            report as soon as the benchmark finishes.
          </p>
        </div>
        <div className="aggregate-hero-actions">
          <Link href="/" className="ghost-btn">
            Back to Dashboard
          </Link>
          <Link href="/runs" className="ghost-btn">
            Browse Existing Runs
          </Link>
        </div>
      </section>

      <UploadZone />
    </main>
  );
}
