import {
  getRunType,
  getRunTypeBadgeClass,
  getRunTypeLabel,
  isDenseRun,
  type RunTypeLike,
} from "../lib/run-type";

export function RunTypeBadge({
  run,
  className,
}: {
  run: RunTypeLike;
  className?: string;
}) {
  const runType = getRunType(run);
  const wrapperClasses = ["run-type-badge-group", className].filter(Boolean).join(" ");
  const primaryBadgeClasses = ["run-type-badge", getRunTypeBadgeClass(runType)]
    .filter(Boolean)
    .join(" ");

  return (
    <span className={wrapperClasses}>
      <span className={primaryBadgeClasses}>{getRunTypeLabel(runType)}</span>
      {isDenseRun(run) ? <span className="run-type-badge badge-dense">Dense</span> : null}
      {run.has_ensemble ? <span className="run-type-badge badge-ensemble">Ensemble</span> : null}
    </span>
  );
}
