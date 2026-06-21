"use client";
import { use, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useJobPolling } from "@/lib/hooks";
import ProgressView from "@/components/ProgressView";

export default function JobPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = use(params);
  const router = useRouter();
  const { data: job, isError } = useJobPolling(jobId);

  useEffect(() => {
    if (job?.status === "done" && job.canonical_id) {
      router.replace(`/p/${job.canonical_id}`);
    }
  }, [job?.status, job?.canonical_id, router]);

  return (
    <main className="mx-auto max-w-xl px-4 py-16">
      {isError && (
        <p style={{ color: "var(--downvote)" }}>Could not reach the server.</p>
      )}
      {job && job.status === "failed" && (
        <div>
          <h1 className="mb-2 text-xl font-bold" style={{ color: "var(--text)" }}>
            Analysis failed
          </h1>
          <p style={{ color: "var(--muted)" }}>
            {job.status_message || "Something went wrong."}
          </p>
        </div>
      )}
      {job && job.status !== "failed" && (
        <div>
          <h1 className="mb-1 text-xl font-bold" style={{ color: "var(--text)" }}>
            Analyzing “{job.product_query}”
          </h1>
          <p className="mb-6 text-sm" style={{ color: "var(--muted)" }}>
            This takes about a minute.
          </p>
          <ProgressView statusMessage={job.status_message} />
        </div>
      )}
    </main>
  );
}
