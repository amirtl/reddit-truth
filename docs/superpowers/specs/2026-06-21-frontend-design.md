# Reddit Truth — Frontend design

**Date:** 2026-06-21
**Status:** Approved

## Goal

A web UI for reddit-truth: submit a product query, watch the pipeline run live,
and read an honest, aspect-by-aspect verdict mined from Reddit. Visual identity
mimics Reddit (OrangeRed accent, vote-colored sentiment, light + dark).

## Decisions (from brainstorming)

- **Live progress: polling** the DRF API (`GET /api/jobs/{id}/` ~every 1.5s).
  No Supabase migration; Realtime can replace the polling hook later.
- **Scope:** landing page + recent-analyses list + the core flow
  (submit → progress → results). Results are URL-addressable (bookmarkable).
- **Visual: Reddit-native, light AND dark with a toggle.** OrangeRed `#FF4500`
  brand; sentiment mapped to Reddit's vote colors (positive = upvote orange,
  negative = downvote blue `#7193FF`); white rounded cards on light gray
  (`#DAE0E6`) / `#1A1A1B` cards on `#030303`; clean sans-serif (Inter).

## Stack & layout

- **Next.js (App Router) + TypeScript + TailwindCSS**, in a `frontend/` directory
  in this repo (monorepo).
- **TanStack Query** for fetching/polling (`refetchInterval`, loading/error states).
- **Next.js `rewrites`** proxy `/api/*` → the Django backend (no CORS config).
  Backend base URL via `NEXT_PUBLIC_API_BASE` / rewrite destination env.
- **Inter** via `next/font` (Reddit Sans stand-in).

## Routes / screens

| Route | Screen | Data |
|---|---|---|
| `/` | Landing: hero + submit bar + Recent analyses grid | `GET /api/products/` |
| `/jobs/[jobId]` | Live progress — stage stepper + bar, polls until terminal | `GET /api/jobs/{id}/` |
| `/p/[canonicalId]` | Results dashboard (bookmarkable) | `GET /api/products/{id}/summaries/` |

**Flow:** submit → `POST /api/jobs/` → `router.push('/jobs/[id]')` → on
`status==="done"` → `router.replace('/p/[canonical_id]')`; on `failed` → inline
error from `status_message`.

## Components (small, single-purpose)

- `ThemeToggle` — light/dark, persisted to `localStorage`, toggles `class` on `html`.
- `SubmitBar` — query input + Analyze button; creates a job and navigates.
- `RecentAnalyses` — grid of product cards linking to `/p/[canonicalId]`.
- `ProgressView` — maps the current `status_message` stage to a stepper +
  progress bar (understanding → scraping → filtering → extracting → clustering →
  quantifying → summarizing).
- `ProductHeader` — product name, comment count, subreddits.
- `AspectCard` — vote column (▲ positive% / ▼ negative%) + orange/blue split bar
  + headline; expandable `detail` and `trend_note`; trend arrow.

## Data layer

- `lib/api.ts` — typed functions `createJob(query)`, `getJob(id)`,
  `getProductSummaries(canonicalId)`, `listProducts()`. Types mirror the DRF
  serializers.
- Hooks: `useJobPolling(id)` (polls `getJob` until `status` is `done`/`failed`,
  then stops), `useProductSummaries(id)`, `useRecentProducts()`.

## Backend additions (bundled into this work)

1. **`GET /api/products/`** — list recently analyzed products (id,
   canonical_name, category, comment_count, generated_at) for the landing grid.
   New `ProductSerializer` + list view + url.
2. **`comment_count`** exposed per product (for "N comments analyzed"). Add a
   `comment_count` field to the `Product` model. Thread the analyzed (filtered)
   comment total through `PipelineResult` so `_persist_result` can store it, and
   surface it in both the products list and the summaries response.

## Visual system

Tailwind theme tokens: `brand #FF4500`, `downvote #7193FF`, light bg `#DAE0E6` /
card `#FFFFFF`, dark bg `#030303` / card `#1A1A1B`, plus border/muted tokens.
Dark mode via Tailwind `class` strategy. Sentiment bar = positive (orange) +
negative (downvote blue) widths from `positive_pct` / `negative_pct`.

## Error & empty states

- Job `failed` → show `status_message` (quota, scrape failure, etc.).
- Empty summaries (`status_message` = "no opinions found") → friendly empty state
  suggesting a more popular product.
- Network/query errors → TanStack Query error UI with retry.

## Testing

Vitest + React Testing Library, proportionate:
- `AspectCard` renders the sentiment split and percentages correctly.
- `ProgressView` maps a stage string to the right active step.
- `useJobPolling` stops polling once status is terminal.

## Out of scope (YAGNI)

Auth/accounts, Supabase Realtime, shareable social cards, pagination of recent
list, server-side rendering of results (client-fetched is fine for MVP).
