"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { createJob } from "@/lib/api";

export default function SubmitBar() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError("");
    try {
      const { job_id } = await createJob(query.trim());
      router.push(`/jobs/${job_id}`);
    } catch {
      setError("Could not start the analysis. Is the server running?");
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="w-full">
      <div className="flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Try: Sony WH-1000XM5"
          className="flex-1 rounded-full border px-4 py-2 outline-none"
          style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--text)" }}
        />
        <button
          type="submit"
          disabled={busy}
          className="rounded-full px-5 py-2 font-bold text-white disabled:opacity-60"
          style={{ background: "var(--brand)" }}
        >
          {busy ? "…" : "Analyze"}
        </button>
      </div>
      {error && (
        <p className="mt-2 text-sm" style={{ color: "var(--downvote)" }}>
          {error}
        </p>
      )}
    </form>
  );
}
