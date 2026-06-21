import type { JobStatus } from "./types";

export const STAGES = [
  "understanding",
  "scraping",
  "filtering",
  "extracting",
  "clustering",
  "quantifying",
  "summarizing",
] as const;

export function isTerminal(status: JobStatus): boolean {
  return status === "done" || status === "failed";
}

export function stageIndex(message: string): number {
  return STAGES.indexOf(message as (typeof STAGES)[number]);
}

export function nextPollInterval(status: JobStatus): number | false {
  return isTerminal(status) ? false : 1500;
}
