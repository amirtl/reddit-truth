import type { Job, JobStatus } from "./types";

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

/**
 * Decide how to poll a job query. Stop polling once the query has errored —
 * otherwise a missing/expired job id (404) is polled forever, spamming the
 * console and the server. Until data arrives, poll at the base interval; after
 * that, defer to the job's status (terminal jobs stop polling).
 *
 * Trade-off: a single transient error also halts polling. The QueryClient's
 * `retry` covers one-off blips within a fetch; beyond that we'd rather stop and
 * show the error than hammer the server, and the user can refresh.
 */
export function jobPollInterval(query: {
  state: { status: "pending" | "error" | "success"; data?: Job };
}): number | false {
  if (query.state.status === "error") return false;
  const job = query.state.data;
  return job ? nextPollInterval(job.status) : 1500;
}
