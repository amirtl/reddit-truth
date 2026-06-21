"use client";
import { useQuery } from "@tanstack/react-query";
import { getJob, getProduct, getProductSummaries, listProducts } from "./api";
import { nextPollInterval } from "./status";

export function useJobPolling(jobId: string) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (q) =>
      q.state.data ? nextPollInterval(q.state.data.status) : 1500,
  });
}

export const useProduct = (id: string) =>
  useQuery({ queryKey: ["product", id], queryFn: () => getProduct(id) });

export const useProductSummaries = (id: string) =>
  useQuery({ queryKey: ["summaries", id], queryFn: () => getProductSummaries(id) });

export const useRecentProducts = () =>
  useQuery({ queryKey: ["products"], queryFn: listProducts });
