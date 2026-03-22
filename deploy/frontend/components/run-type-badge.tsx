import { getRunType, getRunTypeBadgeClass, getRunTypeLabel, type RunTypeLike } from "../lib/run-type";

export function RunTypeBadge({
  run,
  className,
}: {
  run: RunTypeLike;
  className?: string;
}) {
  const runType = getRunType(run);
  const classes = ["run-type-badge", getRunTypeBadgeClass(runType), className]
    .filter(Boolean)
    .join(" ");

  return <span className={classes}>{getRunTypeLabel(runType)}</span>;
}
