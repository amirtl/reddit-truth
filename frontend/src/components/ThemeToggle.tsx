"use client";
import { useSyncExternalStore } from "react";

// The `dark` class on <html> is external state owned by the DOM (set pre-paint
// by the inline script in layout.tsx). `useSyncExternalStore` is the canonical
// way to read external mutable state: it avoids setState-in-effect (no cascading
// renders) and stays hydration-safe by returning a stable server snapshot.

function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  return () => observer.disconnect();
}

const isDark = () => document.documentElement.classList.contains("dark");

export default function ThemeToggle() {
  // Server can't know the user's theme; render light, then the store re-reads
  // the real DOM state on the client (matching the pre-paint script).
  const dark = useSyncExternalStore(subscribe, isDark, () => false);

  function toggle() {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  }

  return (
    <button
      onClick={toggle}
      aria-label="Toggle theme"
      className="rounded-full border px-3 py-1 text-sm"
      style={{ borderColor: "var(--border)", color: "var(--text)" }}
    >
      {dark ? "☀️" : "🌙"}
    </button>
  );
}
