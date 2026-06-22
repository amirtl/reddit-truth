import type { Product } from "@/lib/types";

export default function ProductHeader({ product }: { product: Product }) {
  return (
    <div>
      <h1 className="text-2xl font-bold" style={{ color: "var(--text)" }}>
        {product.canonical_name}
      </h1>
      <p className="text-sm" style={{ color: "var(--muted)" }}>
        {product.comment_count} comments
        {product.subreddits.length > 0 &&
          " · " + product.subreddits.map((s) => "r/" + s).join(" · ")}
      </p>
    </div>
  );
}
