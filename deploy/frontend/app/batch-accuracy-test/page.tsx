import { AccuracyTestPage } from "../../components/accuracy-test-page";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export default function BatchAccuracyTestRoute() {
  return <AccuracyTestPage initialTestMode="batch" />;
}
