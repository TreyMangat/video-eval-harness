import { redirect } from "next/navigation";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
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

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: Promise<{
    run?: string | string[];
    dataDir?: string | string[];
    limit?: string | string[];
  }>;
}) {
  const resolvedSearchParams = await searchParams;
  const runId = readFirst(resolvedSearchParams.run);
  const dataDir = readFirst(resolvedSearchParams.dataDir);
  const limit = readFirst(resolvedSearchParams.limit);

  if (runId) {
    redirect(buildHref(`/report/${runId}`, { dataDir }));
  }

  redirect(buildHref("/", { dataDir, limit }));
}
