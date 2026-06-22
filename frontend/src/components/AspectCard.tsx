"use client";
import { useState } from "react";
import type { AspectSummary } from "@/lib/types";

const TREND: Record<string, string> = { improving: "▴", declining: "▾", stable: "" };

export default function AspectCard({ summary }: { summary: AspectSummary }) {
  const [open, setOpen] = useState(false);
  const pos = Math.round(summary.positive_pct);
  const neg = Math.round(summary.negative_pct);

  return (
    <div
      className="rounded-lg border p-4"
      style={{ background: "var(--card)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center gap-3">
        <div className="w-12 leading-tight">
          <div className="flex items-center gap-1.5">
            <span style={{ color: "var(--brand)" }}>▲</span>
            <span className="text-xs font-bold" style={{ color: "var(--text)" }}>
              {pos}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span style={{ color: "var(--downvote)" }}>▼</span>
            <span className="text-xs font-bold" style={{ color: "var(--text)" }}>
              {neg}
            </span>
          </div>
        </div>
        <div className="flex-1">
          <div className="flex justify-between text-sm" style={{ color: "var(--text)" }}>
            <strong>{summary.aspect}</strong>
            <span style={{ color: "var(--muted)" }}>
              {summary.recent_trend} {TREND[summary.recent_trend]}
            </span>
          </div>
          <div
            className="mt-1 flex h-2 overflow-hidden rounded"
            style={{ background: "var(--track)" }}
          >
            <div data-testid="pos-bar" style={{ width: `${pos}%`, background: "var(--brand)" }} />
            <div data-testid="neg-bar" style={{ width: `${neg}%`, background: "var(--downvote)" }} />
          </div>
          <p className="mt-2 text-sm" style={{ color: "var(--text)" }}>
            {summary.headline}
          </p>
          <button
            onClick={() => setOpen(!open)}
            className="mt-1 text-xs underline"
            style={{ color: "var(--muted)" }}
          >
            {open ? "Hide" : "Details"}
          </button>
          {open && (
            <div className="mt-2 text-sm" style={{ color: "var(--text)" }}>
              <p>{summary.detail}</p>
              {summary.trend_note && (
                <p className="mt-1" style={{ color: "var(--muted)" }}>
                  {summary.trend_note}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
