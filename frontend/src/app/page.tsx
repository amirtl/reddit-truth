import SubmitBar from "@/components/SubmitBar";
import RecentAnalyses from "@/components/RecentAnalyses";
import ThemeToggle from "@/components/ThemeToggle";

export default function Home() {
  return (
    <main className="mx-auto max-w-3xl px-4 py-10">
      <div className="mb-8 flex items-center justify-between">
        <span className="text-xl font-bold" style={{ color: "var(--text)" }}>
          reddit<span style={{ color: "var(--brand)" }}>truth</span>
        </span>
        <ThemeToggle />
      </div>
      <h1 className="mb-2 text-3xl font-bold" style={{ color: "var(--text)" }}>
        What does Reddit really think?
      </h1>
      <p className="mb-6" style={{ color: "var(--muted)" }}>
        Type a product. We mine real Reddit opinions into an honest,
        aspect-by-aspect verdict.
      </p>
      <div className="mb-12">
        <SubmitBar />
      </div>
      <RecentAnalyses />
    </main>
  );
}
