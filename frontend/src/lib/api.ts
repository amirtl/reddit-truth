import type { Job, AspectSummary, Product } from "./types";

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  return res.json() as Promise<T>;
}

export async function createJob(
  query: string
): Promise<{ job_id: string; status: string }> {
  const res = await fetch("/api/jobs/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`submit failed: ${res.status}`);
  return res.json();
}

export const getJob = (id: string) => get<Job>(`/api/jobs/${id}/`);
export const getProduct = (id: string) => get<Product>(`/api/products/${id}/`);
export const getProductSummaries = (id: string) =>
  get<AspectSummary[]>(`/api/products/${id}/summaries/`);
export const listProducts = () => get<Product[]>("/api/products/");
