"use client";

import type { ReactNode } from "react";

export function InfoTooltip({
  text,
  label = "i",
  className = "",
}: {
  text: string;
  label?: ReactNode;
  className?: string;
}) {
  return (
    <span className={`info-tooltip ${className}`.trim()} tabIndex={0}>
      <span className="info-tooltip-trigger" aria-hidden="true">
        {label}
      </span>
      <span className="info-tooltip-popover" role="tooltip">
        {text}
      </span>
    </span>
  );
}
