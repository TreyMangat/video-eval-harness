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
          <h1 className="run-title">Upload a clip and compare models.</h1>
          <p className="chart-desc">
            This comparison mode runs multiple models on the same clip, previews the estimated
            spend live, and routes you to the report as soon as the benchmark finishes.
          </p>
        </div>
        <div className="analysis-actions">
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
