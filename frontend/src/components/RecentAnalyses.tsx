"use client";
import Link from "next/link";
import { useRecentProducts } from "@/lib/hooks";

export default function RecentAnalyses() {
  const { data, isLoading } = useRecentProducts();
  if (isLoading) return <p style={{ color: "var(--muted)" }}>Loading recent…</p>;
  if (!data || data.length === 0) return null;

  return (
    <div>
      <h2 className="mb-3 text-sm font-bold uppercase" style={{ color: "var(--muted)" }}>
        Recent analyses
      </h2>
      <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
        {data.map((p) => (
          <Link
            key={p.id}
            href={`/p/${p.id}`}
            className="rounded-lg border p-4"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--text)" }}
          >
            <div className="font-bold">{p.canonical_name}</div>
            <div className="text-xs" style={{ color: "var(--muted)" }}>
              {p.comment_count} comments · {p.category}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
