import { describe, it, expect } from "vitest";
import {
  isTerminal,
  STAGES,
  stageIndex,
  nextPollInterval,
  jobPollInterval,
} from "./status";
import type { Job } from "./types";

const job = (status: Job["status"]): Job => ({
  id: "abc",
  product_query: "q",
  canonical_id: null,
  status,
  progress: 0,
  status_message: "",
  created_at: "",
  completed_at: null,
});

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

describe("jobPollInterval", () => {
  it("stops polling when the query has errored (e.g. unknown job 404)", () => {
    expect(jobPollInterval({ state: { status: "error", data: undefined } })).toBe(
      false
    );
  });

  it("polls at the base interval before any data arrives", () => {
    expect(jobPollInterval({ state: { status: "pending", data: undefined } })).toBe(
      1500
    );
  });

  it("keeps polling a running job and stops on a terminal one", () => {
    expect(jobPollInterval({ state: { status: "success", data: job("running") } })).toBe(1500);
    expect(jobPollInterval({ state: { status: "success", data: job("done") } })).toBe(false);
    expect(jobPollInterval({ state: { status: "success", data: job("failed") } })).toBe(false);
  });
});
