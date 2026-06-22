"use client";
import { STAGES, stageIndex } from "@/lib/status";

export default function ProgressView({ statusMessage }: { statusMessage: string }) {
  const current = stageIndex(statusMessage);
  return (
    <ol className="space-y-2">
      {STAGES.map((stage, i) => {
        const active = current >= 0 && i <= current;
        return (
          <li
            key={stage}
            data-active={active}
            className="flex items-center gap-3 text-sm"
            style={{ color: active ? "var(--text)" : "var(--muted)" }}
          >
            <span style={{ color: active ? "var(--brand)" : "var(--muted)" }}>
              {active ? "●" : "○"}
            </span>
            {stage}
          </li>
        );
      })}
    </ol>
  );
}
