export type JobStatus = "pending" | "running" | "done" | "failed";
export type Trend = "improving" | "declining" | "stable";

export interface Job {
  id: string;
  product_query: string;
  canonical_id: string | null;
  status: JobStatus;
  progress: number;
  status_message: string;
  created_at: string;
  completed_at: string | null;
}

export interface AspectSummary {
  aspect: string;
  mention_pct: number;
  positive_pct: number;
  negative_pct: number;
  recent_trend: Trend;
  headline: string;
  detail: string;
  trend_note: string;
  generated_at: string;
}

export interface Product {
  id: string;
  canonical_name: string;
  category: string;
  subreddits: string[];
  comment_count: number;
  created_at: string;
}
