# Reddit Truth — System Design Spec
**Date:** 2026-06-15  
**Status:** Approved  
**Author:** Amir Salari

---

## Problem

Amazon reviews are gamed. Reddit has honest opinions — but they're buried across dozens of threads, unstructured, and noisy. There's no way to get the aggregated signal without reading hundreds of comments manually.

**Core value:** Given a product name, aggregate Reddit discussions at scale, extract what real users praise and complain about, quantify it, and surface the honest signal.

**Example output:** "87% of 340 Reddit comments mention battery life. 71% positive — but recent threads flag degradation after 8 months."

---

## Product Decisions

- **Form factor:** Web app (MVP). Browser extension is Phase 2.
- **Entry point:** User types a product name → gets structured Reddit analysis
- **MVP scope:** Single product analysis only. Cross-product comparison is Phase 2.
- **UI:** Deferred. Backend architecture is the MVP priority.

---

## Architecture — Hybrid Pipeline

Seven components in sequence. LLM handles only what requires reasoning. Traditional NLP handles the rest.

```
User Query
    ↓
① Product Understanding Agent  (LLM — resolves query, finds subreddits)
    ↓
② Reddit Scraper               (PRAW — fetches posts + comments)
    ↓
③ Noise Filter                 (rule-based — removes short, off-topic, bot)
    ↓
④ Aspect Extractor             (LLM batched — extracts aspect claims per comment)
    ↓
⑤ Embedder + Clusterer         (sentence-transformers + mean-shift)
    ↓
⑥ Quantifier                   (counts mentions, calculates %, computes trend)
    ↓
⑦ Summarizer                   (LLM — generates headline + detail per cluster)
    ↓
Structured Output → Cache → Frontend
```

---

## Component Interfaces

Each component has exactly one input type and one output type. Swapping the LLM or clustering algorithm does not break other components.

| Component | Input | Output |
|---|---|---|
| ProductUnderstandingAgent | `raw_query: str` | `ProductInfo(canonical_id, canonical_name, category, search_terms[], subreddits[])` |
| RedditScraper | `ProductInfo` | `List[RawComment(id, text, score, created_at, subreddit, post_url)]` |
| NoiseFilter | `List[RawComment]` | `List[RawComment]` — min 10 words, on-topic |
| AspectExtractor | `List[RawComment]` (batched 15/call) | `List[AspectClaim(comment_id, aspect, sentiment, quote)]` |
| EmbedderClusterer | `List[AspectClaim]` | `List[Cluster(label, claims[], positive_count, negative_count)]` |
| Quantifier | `List[Cluster]`, `List[RawComment]` (for timestamps + total count) | `List[QuantifiedAspect(label, mention_pct, positive_pct, negative_pct, recent_trend)]` |
| Summarizer | `QuantifiedAspect` + `sample_quotes[]` | `AspectSummary(headline, detail, trend_note)` |

---

## Trend Calculation

Compare last 90 days vs all-time baseline per aspect:

```
delta = recent_positive_pct - baseline_positive_pct

delta > +10%  → "improving"
delta < -10%  → "declining"
else          → "stable"
```

Relative comparison, not absolute thresholds. A product at 60% that was 80% is declining even though 60% is technically positive.

---

## Caching — 3 Layers

All layers keyed on `canonical_id` produced by the Product Understanding Agent.

| Layer | Storage | TTL | Contents |
|---|---|---|---|
| Query → canonical_id | Redis | 30 days | Maps raw queries to canonical product IDs |
| Final summaries | Redis | 7 days | Complete structured output, served in <100ms |
| Clusters + counts | PostgreSQL + Redis | 24h | Intermediate NLP output |
| Raw comments | PostgreSQL | 24h | Raw Reddit data |

**Two-step lookup:**
1. Raw query → Redis `q:{query}` → canonical_id (or call Product Agent)
2. canonical_id → Redis `product:{id}:summary` → hit or miss → deeper layers

---

## Data Models (PostgreSQL via Supabase)

```sql
products (
    id VARCHAR PK,          -- "sony-wh-1000xm5"
    canonical_name VARCHAR,
    category VARCHAR,
    search_terms JSONB,
    subreddits JSONB,
    created_at TIMESTAMP
)

jobs (
    id VARCHAR PK,          -- job_id returned to frontend
    product_query VARCHAR,
    canonical_id VARCHAR,
    status VARCHAR,         -- pending / running / done / failed
    progress INTEGER,       -- 0–100
    status_message VARCHAR, -- "Extracting opinions from 240 comments..."
    created_at TIMESTAMP,
    completed_at TIMESTAMP
)

raw_comments (
    id VARCHAR PK,
    product_id FK → products,
    text TEXT,
    score INTEGER,
    subreddit VARCHAR,
    post_url VARCHAR,
    created_at TIMESTAMP
)

aspect_claims (
    id SERIAL PK,
    comment_id FK → raw_comments,
    product_id FK → products,
    aspect VARCHAR,
    sentiment VARCHAR,      -- positive / negative / mixed
    quote TEXT
)

aspect_summaries (
    id SERIAL PK,
    product_id FK → products,
    aspect VARCHAR,
    mention_pct FLOAT,
    positive_pct FLOAT,
    negative_pct FLOAT,
    recent_trend VARCHAR,   -- improving / declining / stable
    headline TEXT,
    detail TEXT,
    generated_at TIMESTAMP
)

query_cache (
    raw_query VARCHAR PK,
    canonical_id FK → products,
    created_at TIMESTAMP
)
```

---

## Async Pipeline — Celery + Supabase Realtime

The pipeline takes ~45 seconds. HTTP requests cannot block for 45 seconds. Solution: task queue.

**Flow:**
1. `POST /api/analyze/` → Celery `.delay()` → returns `{job_id}` in <50ms
2. Celery worker picks up task, runs pipeline, writes `UPDATE jobs SET status=..., progress=...` at each stage
3. Supabase Realtime detects each UPDATE → pushes to frontend WebSocket
4. On `status=done`, frontend fetches `GET /api/products/{canonical_id}/`
5. Fallback: if WebSocket fails, frontend polls `GET /api/jobs/{job_id}/` every 3s

---

## API Endpoints (DRF)

| Method | Endpoint | Description | Response |
|---|---|---|---|
| POST | `/api/analyze/` | Submit query, start pipeline | `{job_id, cached: bool, canonical_id: str\|null}` |
| GET | `/api/jobs/{job_id}/` | Fallback polling | `{status, progress, message, canonical_id}` |
| GET | `/api/products/{canonical_id}/` | Get full analysis | `{canonical_id, canonical_name, comment_count, aspects[]}` |

---

## LLM Configuration

All LLM calls go through **LiteLLM** — a unified interface supporting any provider. Model selection is in `config.yml`, zero code changes to swap.

```yaml
llms:
  product_understanding: "ollama/llama3.2"       # local, free
  aspect_extraction:     "gemini/gemini-2.0-flash" # free tier (1,500 req/day)
  summarization:         "ollama/llama3.2"       # local, free

embeddings:
  provider: "local"
  model: "all-MiniLM-L6-v2"   # sentence-transformers, no API, free
```

**Supported providers (drop-in via config):** Ollama (local), Gemini, OpenAI, Anthropic, and 100+ more.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend API | Django + DRF | Career targeting — Django/DRF/Celery in demand at senior level |
| Task queue | Celery + Redis broker | Industry standard for async tasks, appears in enterprise job reqs |
| Database | Supabase (PostgreSQL) | Hosted, free tier, Realtime built-in |
| Realtime push | Supabase Realtime | WebSocket push on DB changes — no polling needed |
| Cache | Redis | Fast TTL cache + Celery broker (same instance) |
| Frontend | Next.js + TailwindCSS | SSR, SEO, Supabase JS client |
| LLM abstraction | LiteLLM | Swap any provider via config |
| Embeddings | sentence-transformers | Local, free, good enough for MVP |
| Clustering | scikit-learn (mean-shift) | K auto-detected, correct for unknown number of aspects |
| Reddit | PRAW | Official Python Reddit API wrapper |

**Total MVP running cost: $0/month** (all free tiers + local Ollama)

---

## What's Out of Scope for MVP

- User accounts / authentication
- Cross-product comparison (Phase 2)
- Browser extension (Phase 2)
- Pre-computed product catalog (Phase 2 — only after validation)
- UI design polish (deferred — backend backbone first)
