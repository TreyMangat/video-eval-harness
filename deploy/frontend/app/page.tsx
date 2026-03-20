import { BenchmarkDashboard } from "../components/benchmark-dashboard";

export default function HomePage() {
  return <BenchmarkDashboard dataDir={process.env.VBENCH_RUNS_DIR} />;
}
