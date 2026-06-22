# Reddit Truth

> Amazon reviews are gamed. Reddit has the honest signal — buried in hundreds of threads across dozens of subreddits. **Reddit Truth** extracts it.

Given a product name, it scrapes Reddit at scale, extracts aspect-level opinions from real users, clusters them semantically, quantifies sentiment with citations, and surfaces a structured analysis in seconds.

**Example output:**
> *"87% of 340 Reddit comments mention battery life. 71% positive — but recent threads flag degradation after 8 months."*

---

## How it works

```
User query: "Sony WH-1000XM5"
        │
        ▼
┌─────────────────────────┐
│  Product Understanding  │  LLM resolves query → canonical product + subreddits
│  Agent                  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Reddit Scraper         │  PRAW fetches posts + comments across subreddits
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Noise Filter           │  Removes bots, short comments, spam
└──────────┬──────────────┘
           │  ~240 clean comments
           ▼
┌─────────────────────────┐
│  Aspect Extractor       │  LLM (Gemini Flash, batched) → aspect claims per comment
│                         │  "Battery is great but ANC disappoints" → 2 claims
└──────────┬──────────────┘
           │  ~580 aspect claims
           ▼
┌─────────────────────────┐
│  Embedder + Clusterer   │  sentence-transformers + mean-shift → auto-discover topics
└──────────┬──────────────┘
           │  8 clusters: battery, ANC, comfort, durability...
           ▼
┌─────────────────────────┐
│  Quantifier             │  Count mentions, compute %, detect 90-day trend
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Summarizer             │  LLM generates headline + detail per cluster
└──────────┬──────────────┘
           │
           ▼
   Structured output → cached → served via REST API
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | Django 4.2 + Django REST Framework |
| Async tasks | Celery + Redis |
| Database | Supabase (PostgreSQL) + Supabase Realtime |
| LLM abstraction | LiteLLM — swap any provider via `config.yml` |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, local, free) |
| Clustering | scikit-learn Mean-Shift |
| Reddit | PRAW (Python Reddit API Wrapper) |
| Config & validation | Pydantic + pydantic-settings |
| Frontend | Next.js (coming soon) |

**Default MVP cost: $0/month** — Ollama (local) + Gemini Flash free tier + Supabase free tier + Redis free tier.

---

## Quick Start

### With Docker (recommended)

```bash
git clone https://github.com/amirtl/reddit-truth.git
cd reddit-truth
cp .env.example .env      # fill in your credentials
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Local development

```bash
git clone https://github.com/amirtl/reddit-truth.git
cd reddit-truth
make install              # creates venv + installs dependencies via uv
# edit .env with your credentials
make migrate
make run                  # Django dev server
make worker               # Celery worker (separate terminal)
```

---

## Configuration

All LLM providers are configured in `config.yml` — no code changes needed to swap models:

```yaml
llms:
  product_understanding: "ollama/llama3.2"       # local, free
  aspect_extraction: "gemini/gemini-2.0-flash"   # free tier (1,500 req/day)
  summarization: "ollama/llama3.2"               # local, free

embeddings:
  provider: "local"
  model: "all-MiniLM-L6-v2"
```

**Supported providers** (via LiteLLM): Ollama, Gemini, OpenAI, Anthropic, and 100+ more.

Environment variables are validated at startup via Pydantic Settings. See `.env.example` for all options.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/jobs/` | Submit a product query → dispatches the pipeline |
| `GET` | `/api/jobs/{id}/` | Poll job status / progress |
| `GET` | `/api/products/{canonical_id}/summaries/` | Fetch the aspect summaries |

**Example:**

```bash
# Submit a query (returns 202 Accepted — work runs in the background)
curl -X POST http://localhost:8000/api/jobs/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Sony WH-1000XM5"}'

# Response
{"job_id": "a3f9c2e1...", "status": "pending"}

# Poll status until done/failed
curl http://localhost:8000/api/jobs/a3f9c2e1.../
{"status": "running", "progress": 0, "status_message": "extracting", ...}

# Fetch the summaries once the job's canonical_id is set
curl http://localhost:8000/api/products/sony-wh-1000xm5/summaries/
```

---

## Running Tests

```bash
make test           # full suite
make test-fast      # skip slow embedding tests
```

Tests run against SQLite in-memory (no Supabase needed). Celery tasks run synchronously in tests.

---

## Smoke Test (end-to-end)

The unit suite mocks every external service. The smoke test mocks **nothing** —
it drives the real stack through the HTTP API to prove the pipes connect and a
real LLM produces well-formed output.

**1. Bring up the dependencies** (Postgres, Redis, an LLM backend, Reddit creds):

```bash
cp -n .env.example .env      # fill in REDDIT_* and, if used, GEMINI_API_KEY / DB_*
ollama serve &               # local LLM (model from config.yml)
redis-server &               # broker
make migrate                 # create tables
```

**2. Start the server and a worker** (separate terminals):

```bash
make run        # Django at http://localhost:8000
make worker     # Celery worker
```

**3. Run the smoke test:**

```bash
make smoke                       # default query (Sony WH-1000XM5)
make smoke QUERY="AirPods Pro"   # or pick your own
```

It submits the query, prints each stage as the job moves `pending → running →
done`, then validates the returned summaries' **shape** (required fields,
percentage ranges, valid trend values). Exit code `0` = healthy. It does not
assert exact text — an LLM is non-deterministic.

---

## Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000  (proxies /api to the Django backend)
```

Needs the backend running (`make run`) and a worker (`make worker`). Submit a
product, watch the live pipeline progress, then read the aspect-by-aspect verdict.
Light/dark toggle in the corner. Built with Next.js + TailwindCSS + TanStack Query.

---

## Project Structure

```
reddit-truth/
├── config.yml              # LLM provider config (swap models here)
├── pipeline/               # NLP pipeline — no Django dependency
│   ├── types.py            # Pydantic data models
│   ├── config.py           # Config loader
│   ├── product_agent.py    # Component 1: query → canonical product
│   ├── scraper.py          # Component 2: Reddit scraping
│   ├── noise_filter.py     # Component 3: comment filtering
│   ├── aspect_extractor.py # Component 4: LLM aspect extraction
│   ├── embedder_clusterer.py # Component 5: semantic clustering
│   ├── quantifier.py       # Component 6: quantification + trend
│   ├── summarizer.py       # Component 7: LLM summaries
│   └── runner.py           # Orchestrator
├── core/                   # Django app: database models
├── api/                    # Django app: REST endpoints
├── tasks/                  # Celery tasks
└── docs/                   # Design spec + implementation plan
```

---

## License

MIT
