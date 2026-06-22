import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import AspectCard from "./AspectCard";
import type { AspectSummary } from "@/lib/types";

const summary: AspectSummary = {
  aspect: "battery life",
  mention_pct: 71,
  positive_pct: 29,
  negative_pct: 71,
  recent_trend: "declining",
  headline: "Battery fades over time",
  detail: "Users report degradation.",
  trend_note: "Worse recently.",
  generated_at: "2026-06-21T00:00:00Z",
};

describe("AspectCard", () => {
  it("shows the aspect, headline, and rounded sentiment numbers", () => {
    render(<AspectCard summary={summary} />);
    expect(screen.getByText("battery life")).toBeInTheDocument();
    expect(screen.getByText("Battery fades over time")).toBeInTheDocument();
    expect(screen.getByText("29")).toBeInTheDocument(); // upvote (positive)
    expect(screen.getByText("71")).toBeInTheDocument(); // downvote (negative)
  });

  it("sizes the positive bar to positive_pct", () => {
    const { container } = render(<AspectCard summary={summary} />);
    const pos = container.querySelector('[data-testid="pos-bar"]') as HTMLElement;
    expect(pos.style.width).toBe("29%");
  });
});
