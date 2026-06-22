"use client";
import { use } from "react";
import Link from "next/link";
import { useProduct, useProductSummaries } from "@/lib/hooks";
import ProductHeader from "@/components/ProductHeader";
import AspectCard from "@/components/AspectCard";
import ThemeToggle from "@/components/ThemeToggle";

export default function ResultsPage({ params }: { params: Promise<{ canonicalId: string }> }) {
  const { canonicalId } = use(params);
  const { data: product } = useProduct(canonicalId);
  const { data: summaries, isLoading } = useProductSummaries(canonicalId);

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <div className="mb-6 flex items-center justify-between">
        <Link href="/" className="text-sm font-bold" style={{ color: "var(--text)" }}>
          ← reddit<span style={{ color: "var(--brand)" }}>truth</span>
        </Link>
        <ThemeToggle />
      </div>

      {product && <ProductHeader product={product} />}

      <div className="mt-6 space-y-3">
        {isLoading && <p style={{ color: "var(--muted)" }}>Loading…</p>}
        {summaries && summaries.length === 0 && (
          <p style={{ color: "var(--muted)" }}>
            No opinions found — try a more popular product or broader terms.
          </p>
        )}
        {summaries?.map((s) => (
          <AspectCard key={s.aspect} summary={s} />
        ))}
      </div>
    </main>
  );
}
