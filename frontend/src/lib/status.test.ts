import { describe, it, expect } from "vitest";
import { isTerminal, STAGES, stageIndex, nextPollInterval } from "./status";

describe("status helpers", () => {
  it("treats done and failed as terminal", () => {
    expect(isTerminal("done")).toBe(true);
    expect(isTerminal("failed")).toBe(true);
    expect(isTerminal("running")).toBe(false);
  });

  it("maps a stage message to its index", () => {
    expect(stageIndex("scraping")).toBe(STAGES.indexOf("scraping"));
    expect(stageIndex("unknown")).toBe(-1);
  });

  it("stops polling once terminal", () => {
    expect(nextPollInterval("running")).toBe(1500);
    expect(nextPollInterval("done")).toBe(false);
    expect(nextPollInterval("failed")).toBe(false);
  });
});
