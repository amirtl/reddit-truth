import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ProgressView from "./ProgressView";

describe("ProgressView", () => {
  it("marks stages up to the current one as active", () => {
    render(<ProgressView statusMessage="extracting" />);
    const scraping = screen.getByText("scraping").closest("li")!;
    const clustering = screen.getByText("clustering").closest("li")!;
    expect(scraping.getAttribute("data-active")).toBe("true");
    expect(clustering.getAttribute("data-active")).toBe("false");
  });
});
