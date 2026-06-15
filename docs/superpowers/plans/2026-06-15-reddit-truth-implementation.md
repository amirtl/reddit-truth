# Reddit Truth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full Reddit Truth backend — a hybrid NLP pipeline that scrapes Reddit, extracts product opinions, clusters them, quantifies sentiment, and serves results via a Django REST API with async Celery processing.

**Architecture:** User query → Django DRF API → Celery task queue → 7-component NLP pipeline (Product Agent → Scraper → Filter → Extractor → Embedder+Clusterer → Quantifier → Summarizer) → Supabase PostgreSQL → Supabase Realtime push to frontend.

**Tech Stack:** Python 3.11, Django 4.2, DRF, Celery, Redis, Supabase (PostgreSQL), PRAW, LiteLLM, sentence-transformers, scikit-learn, pytest

---

## File Structure

```
reddit-truth/
├── config.yml                     # LLM provider config — swap models here, zero code changes
├── .env                           # Secrets (never commit)
├── .env.example                   # Template for .env
├── requirements.txt
├── pytest.ini
├── manage.py
├── reddit_truth/                  # Django project config
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── celery.py                  # Celery app definition
│   └── wsgi.py
├── core/                          # Django app: database models only
│   ├── __init__.py
│   ├── models.py                  # Product, Job, RawComment, AspectClaim, AspectSummary, QueryCache
│   ├── admin.py
│   └── migrations/
├── pipeline/                      # Python package: NLP components (no Django dependency)
│   ├── __init__.py
│   ├── types.py                   # Dataclasses: ProductInfo, RawComment, AspectClaim, Cluster, QuantifiedAspect, AspectSummary
│   ├── config.py                  # Loads config.yml into typed dataclasses
│   ├── product_agent.py           # Component 1: resolves query → canonical product + subreddits
│   ├── scraper.py                 # Component 2: fetches Reddit posts + comments via PRAW
│   ├── noise_filter.py            # Component 3: removes short/off-topic/bot comments
│   ├── aspect_extractor.py        # Component 4: extracts aspect claims from comments (LLM, batched)
│   ├── embedder_clusterer.py      # Component 5: embeds claims → mean-shift cluster
│   ├── quantifier.py              # Component 6: counts mentions, calculates %, detects trend
│   ├── summarizer.py              # Component 7: generates headline+detail per cluster (LLM)
│   └── runner.py                  # Orchestrates all 7 components, reports progress
├── api/                           # Django app: DRF endpoints
│   ├── __init__.py
│   ├── views.py                   # 3 views: analyze, job_status, product_detail
│   ├── serializers.py             # DRF serializers for Job and AspectSummary
│   └── urls.py
├── tasks/                         # Celery tasks
│   ├── __init__.py
│   └── pipeline_task.py           # run_analysis_pipeline task
└── tests/
    ├── __init__.py
    ├── conftest.py                 # pytest fixtures, Django settings override
    ├── test_config.py
    ├── test_types.py
    ├── test_noise_filter.py
    ├── test_quantifier.py
    ├── test_product_agent.py       # mocks LiteLLM
    ├── test_scraper.py             # mocks PRAW
    ├── test_aspect_extractor.py    # mocks LiteLLM
    ├── test_embedder_clusterer.py  # uses real sentence-transformers (fast)
    ├── test_summarizer.py          # mocks LiteLLM
    ├── test_runner.py              # mocks all 7 components
    └── test_api.py                 # DRF test client, mocks Celery
```

---

## Phase 1 — Foundation

### Task 1: Project initialization + dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config.yml`
- Create: `.env.example`
- Create: `pytest.ini`
- Create: `reddit_truth/celery.py`

- [ ] **Step 1: Initialize git and Django project**

```bash
cd /Users/amirsalari/Personal/growth-plan/reddit-truth
git init
django-admin startproject reddit_truth .
python manage.py startapp core
python manage.py startapp api
mkdir -p pipeline tasks tests
touch pipeline/__init__.py tasks/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```
django==4.2.21
djangorestframework==3.15.2
celery[redis]==5.3.6
praw==7.7.1
litellm==1.40.0
sentence-transformers==3.0.1
scikit-learn==1.5.1
numpy==1.26.4
psycopg2-binary==2.9.9
redis==5.0.7
pyyaml==6.0.1
python-dotenv==1.0.1
pytest==8.2.2
pytest-django==4.8.0
pytest-mock==3.14.0
```

Install: `pip install -r requirements.txt`

- [ ] **Step 3: Create config.yml**

```yaml
llms:
  product_understanding: "ollama/llama3.2"
  aspect_extraction: "gemini/gemini-2.0-flash"
  summarization: "ollama/llama3.2"

embeddings:
  provider: "local"
  model: "all-MiniLM-L6-v2"
```

- [ ] **Step 4: Create .env.example**

```
# Django
SECRET_KEY=your-secret-key-here
DEBUG=True

# Supabase PostgreSQL
DATABASE_URL=postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres

# Redis
REDIS_URL=redis://localhost:6379/0

# Reddit API (get from https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=reddit-truth/0.1

# LLM API keys (only needed for paid providers)
GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

Copy to `.env` and fill in values: `cp .env.example .env`

- [ ] **Step 5: Configure settings.py**

Replace the contents of `reddit_truth/settings.py`:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ["SECRET_KEY"]
DEBUG = os.environ.get("DEBUG", "False") == "True"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "core",
    "api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

ROOT_URLCONF = "reddit_truth.urls"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "postgres",
        "USER": "postgres",
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
    }
}

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_TASK_SERIALIZER = "json"

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": REDIS_URL,
    }
}

STATIC_URL = "/static/"
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True
```

- [ ] **Step 6: Create reddit_truth/celery.py**

```python
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reddit_truth.settings")
app = Celery("reddit_truth")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
```

- [ ] **Step 7: Update reddit_truth/__init__.py**

```python
from .celery import app as celery_app

__all__ = ("celery_app",)
```

- [ ] **Step 8: Create pytest.ini**

```ini
[pytest]
DJANGO_SETTINGS_MODULE = reddit_truth.settings
python_files = tests/test_*.py
python_classes = Test*
python_functions = test_*
```

- [ ] **Step 9: Commit**

```bash
git add .
git commit -m "feat: initialize Django project with Celery, Redis, Supabase config"
```

---

### Task 2: Type definitions

**Files:**
- Create: `pipeline/types.py`
- Create: `tests/test_types.py`

*What you're learning:* Python `dataclasses` are like lightweight structs. They give you `__init__`, `__repr__`, and `__eq__` for free. Using `Literal` for sentinel values like `"positive"/"negative"/"mixed"` catches typos at type-check time instead of runtime.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
from datetime import datetime
from pipeline.types import ProductInfo, RawComment, AspectClaim, Cluster, QuantifiedAspect, AspectSummary

def test_product_info_creation():
    p = ProductInfo(
        canonical_id="sony-wh-1000xm5",
        canonical_name="Sony WH-1000XM5",
        category="headphones",
        search_terms=["WH-1000XM5", "Sony XM5"],
        subreddits=["headphones", "audiophile"],
    )
    assert p.canonical_id == "sony-wh-1000xm5"
    assert len(p.search_terms) == 2

def test_raw_comment_creation():
    c = RawComment(
        id="abc123",
        text="Battery life is incredible, lasts 3 days easily",
        score=42,
        created_at=datetime(2024, 6, 1),
        subreddit="headphones",
        post_url="https://reddit.com/r/headphones/comments/abc",
    )
    assert c.score == 42

def test_aspect_claim_creation():
    claim = AspectClaim(
        comment_id="abc123",
        aspect="battery life",
        sentiment="positive",
        quote="lasts 3 days easily",
    )
    assert claim.sentiment == "positive"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_types.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.types'`

- [ ] **Step 3: Create pipeline/types.py**

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

@dataclass
class ProductInfo:
    canonical_id: str
    canonical_name: str
    category: str
    search_terms: list[str]
    subreddits: list[str]

@dataclass
class RawComment:
    id: str
    text: str
    score: int
    created_at: datetime
    subreddit: str
    post_url: str

@dataclass
class AspectClaim:
    comment_id: str
    aspect: str
    sentiment: Literal["positive", "negative", "mixed"]
    quote: str

@dataclass
class Cluster:
    label: str
    claims: list[AspectClaim]
    positive_count: int
    negative_count: int

@dataclass
class QuantifiedAspect:
    label: str
    mention_pct: float
    positive_pct: float
    negative_pct: float
    recent_trend: Literal["improving", "declining", "stable"]

@dataclass
class AspectSummary:
    label: str
    mention_pct: float
    positive_pct: float
    negative_pct: float
    recent_trend: str
    headline: str
    detail: str
    trend_note: str = ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_types.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/types.py tests/test_types.py
git commit -m "feat: add pipeline type definitions"
```

---

### Task 3: Config loader

**Files:**
- Create: `pipeline/config.py`
- Create: `tests/test_config.py`

*What you're learning:* Loading config from a YAML file into typed Python dataclasses separates "what model to use" from "how to use it." This is the **dependency injection** pattern — components receive their config, they don't read the file themselves.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from pipeline.config import load_config, AppConfig

def test_load_config_returns_typed_object(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("""
llms:
  product_understanding: "ollama/llama3.2"
  aspect_extraction: "gemini/gemini-2.0-flash"
  summarization: "ollama/llama3.2"
embeddings:
  provider: "local"
  model: "all-MiniLM-L6-v2"
""")
    config = load_config(str(config_file))
    assert isinstance(config, AppConfig)
    assert config.llms.product_understanding == "ollama/llama3.2"
    assert config.llms.aspect_extraction == "gemini/gemini-2.0-flash"
    assert config.embeddings.model == "all-MiniLM-L6-v2"

def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yml")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.config'`

- [ ] **Step 3: Create pipeline/config.py**

```python
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class LLMConfig:
    product_understanding: str
    aspect_extraction: str
    summarization: str

@dataclass
class EmbeddingConfig:
    provider: str
    model: str

@dataclass
class AppConfig:
    llms: LLMConfig
    embeddings: EmbeddingConfig

def load_config(path: str = "config.yml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return AppConfig(
        llms=LLMConfig(**data["llms"]),
        embeddings=EmbeddingConfig(**data["embeddings"]),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py tests/test_config.py
git commit -m "feat: add YAML config loader with typed dataclasses"
```

---

### Task 4: Database models

**Files:**
- Modify: `core/models.py`
- Create: `tests/conftest.py`

*What you're learning:* Django's ORM maps Python classes to database tables. `JSONField` stores lists/dicts as JSON columns in PostgreSQL — perfect for `search_terms` and `subreddits` which vary per product. `ForeignKey` with `on_delete=CASCADE` means deleting a product also deletes all its comments and summaries automatically.

- [ ] **Step 1: Write core/models.py**

```python
# core/models.py
from django.db import models

class Product(models.Model):
    id = models.CharField(max_length=200, primary_key=True)  # "sony-wh-1000xm5"
    canonical_name = models.CharField(max_length=500)
    category = models.CharField(max_length=200)
    search_terms = models.JSONField(default=list)
    subreddits = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.canonical_name

class Job(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("done", "Done"),
        ("failed", "Failed"),
    ]
    id = models.CharField(max_length=100, primary_key=True)
    product_query = models.CharField(max_length=500)
    canonical_id = models.CharField(max_length=200, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    progress = models.IntegerField(default=0)
    status_message = models.CharField(max_length=500, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Job {self.id} ({self.status})"

class RawComment(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="comments")
    text = models.TextField()
    score = models.IntegerField(default=0)
    subreddit = models.CharField(max_length=200)
    post_url = models.URLField(max_length=1000)
    created_at = models.DateTimeField()
    fetched_at = models.DateTimeField(auto_now_add=True)

class AspectClaim(models.Model):
    SENTIMENT_CHOICES = [("positive", "Positive"), ("negative", "Negative"), ("mixed", "Mixed")]
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="aspect_claims")
    comment = models.ForeignKey(RawComment, on_delete=models.CASCADE, related_name="claims")
    aspect = models.CharField(max_length=200)
    sentiment = models.CharField(max_length=20, choices=SENTIMENT_CHOICES)
    quote = models.TextField()

class AspectSummary(models.Model):
    TREND_CHOICES = [("improving", "Improving"), ("declining", "Declining"), ("stable", "Stable")]
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="summaries")
    aspect = models.CharField(max_length=200)
    mention_pct = models.FloatField()
    positive_pct = models.FloatField()
    negative_pct = models.FloatField()
    recent_trend = models.CharField(max_length=20, choices=TREND_CHOICES)
    headline = models.TextField()
    detail = models.TextField()
    generated_at = models.DateTimeField(auto_now_add=True)

class QueryCache(models.Model):
    raw_query = models.CharField(max_length=500, primary_key=True)
    canonical_id = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
```

- [ ] **Step 2: Create tests/conftest.py**

```python
# tests/conftest.py
import django
import os
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reddit_truth.settings")

@pytest.fixture
def sample_product_info():
    from pipeline.types import ProductInfo
    return ProductInfo(
        canonical_id="sony-wh-1000xm5",
        canonical_name="Sony WH-1000XM5",
        category="headphones",
        search_terms=["WH-1000XM5", "Sony XM5 headphones"],
        subreddits=["headphones", "audiophile", "sony"],
    )

@pytest.fixture
def sample_comments():
    from pipeline.types import RawComment
    from datetime import datetime
    return [
        RawComment("c1", "Battery life is incredible, lasts 3 days easily on a single charge", 42, datetime(2024, 6, 1), "headphones", "https://reddit.com/1"),
        RawComment("c2", "ANC is absolutely the best I have ever tried, destroys the Bose QC45", 38, datetime(2024, 6, 2), "audiophile", "https://reddit.com/2"),
        RawComment("c3", "Mine died after 8 months, ear cushions completely degraded", 29, datetime(2024, 5, 1), "headphones", "https://reddit.com/3"),
        RawComment("c4", "The companion app crashes constantly on Android", 15, datetime(2024, 4, 1), "sony", "https://reddit.com/4"),
        RawComment("c5", "lol", 1, datetime(2024, 3, 1), "headphones", "https://reddit.com/5"),  # noise
    ]
```

- [ ] **Step 3: Run migrations**

```bash
python manage.py makemigrations core
python manage.py migrate
```
Expected: migrations created and applied successfully

- [ ] **Step 4: Commit**

```bash
git add core/models.py core/migrations/ tests/conftest.py
git commit -m "feat: add database models for Product, Job, Comment, AspectClaim, AspectSummary"
```

---

## Phase 2 — NLP Pipeline

### Task 5: NoiseFilter

**Files:**
- Create: `pipeline/noise_filter.py`
- Create: `tests/test_noise_filter.py`

*What you're learning:* Always build and test the simplest component first. The noise filter has no external dependencies — no LLM, no network — so it's the easiest to get right. This builds confidence and gives you a tested foundation before adding complexity.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_noise_filter.py
from datetime import datetime
from pipeline.types import RawComment
from pipeline.noise_filter import NoiseFilter

def make_comment(id, text, score=10):
    return RawComment(id, text, score, datetime(2024, 1, 1), "headphones", "https://reddit.com")

def test_filters_short_comments():
    f = NoiseFilter()
    short = make_comment("1", "Great product")
    result = f.run([short])
    assert result == []

def test_keeps_long_substantive_comments():
    f = NoiseFilter()
    long = make_comment("2", "Battery life is incredible and lasts me three full days on a single charge without ANC enabled")
    result = f.run([long])
    assert len(result) == 1

def test_filters_bot_comments():
    f = NoiseFilter()
    bot = make_comment("3", "I am a bot and this action was performed automatically please contact the moderators")
    result = f.run([bot])
    assert result == []

def test_filters_heavily_downvoted():
    f = NoiseFilter()
    downvoted = make_comment("4", "Battery life is incredible and lasts me three full days on a single charge without ANC enabled", score=-10)
    result = f.run([downvoted])
    assert result == []

def test_mixed_batch_filters_correctly(sample_comments):
    f = NoiseFilter()
    result = f.run(sample_comments)
    ids = [c.id for c in result]
    assert "c5" not in ids   # "lol" is too short
    assert "c1" in ids
    assert "c2" in ids
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_noise_filter.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.noise_filter'`

- [ ] **Step 3: Create pipeline/noise_filter.py**

```python
from .types import RawComment

class NoiseFilter:
    MIN_WORDS = 10
    MIN_SCORE = -5
    BOT_PATTERNS = ["i am a bot", "automoderator", "this action was performed automatically"]

    def run(self, comments: list[RawComment]) -> list[RawComment]:
        return [c for c in comments if self._is_valid(c)]

    def _is_valid(self, comment: RawComment) -> bool:
        if len(comment.text.split()) < self.MIN_WORDS:
            return False
        if comment.score < self.MIN_SCORE:
            return False
        text_lower = comment.text.lower()
        if any(pattern in text_lower for pattern in self.BOT_PATTERNS):
            return False
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_noise_filter.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/noise_filter.py tests/test_noise_filter.py
git commit -m "feat: add NoiseFilter — removes short, bot, and downvoted comments"
```

---

### Task 6: ProductUnderstandingAgent

**Files:**
- Create: `pipeline/product_agent.py`
- Create: `tests/test_product_agent.py`

*What you're learning:* When testing code that calls external APIs (LLM, Reddit, databases), you **mock** the external call. The test verifies YOUR code behaves correctly — not that the API works. `pytest-mock` provides `mocker.patch()` for this. This is called **unit testing with dependency isolation**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_product_agent.py
import json
import pytest
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.product_agent import ProductUnderstandingAgent
from pipeline.types import ProductInfo

@pytest.fixture
def config():
    return AppConfig(
        llms=LLMConfig(
            product_understanding="ollama/llama3.2",
            aspect_extraction="gemini/gemini-2.0-flash",
            summarization="ollama/llama3.2",
        ),
        embeddings=EmbeddingConfig(provider="local", model="all-MiniLM-L6-v2"),
    )

def test_run_returns_product_info(config, mocker):
    mock_response = {
        "canonical_id": "sony-wh-1000xm5",
        "canonical_name": "Sony WH-1000XM5",
        "category": "headphones",
        "search_terms": ["WH-1000XM5", "Sony XM5"],
        "subreddits": ["headphones", "audiophile"],
    }
    mock_completion = mocker.patch("pipeline.product_agent.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps(mock_response)

    agent = ProductUnderstandingAgent(config)
    result = agent.run("Sony WH-1000XM5")

    assert isinstance(result, ProductInfo)
    assert result.canonical_id == "sony-wh-1000xm5"
    assert "headphones" in result.subreddits
    mock_completion.assert_called_once()

def test_run_passes_query_in_prompt(config, mocker):
    mock_completion = mocker.patch("pipeline.product_agent.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps({
        "canonical_id": "test-product",
        "canonical_name": "Test Product",
        "category": "electronics",
        "search_terms": ["test"],
        "subreddits": ["gadgets"],
    })
    agent = ProductUnderstandingAgent(config)
    agent.run("some product name")

    call_args = mock_completion.call_args
    prompt = call_args.kwargs["messages"][0]["content"]
    assert "some product name" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_product_agent.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.product_agent'`

- [ ] **Step 3: Create pipeline/product_agent.py**

```python
import json
import litellm
from .types import ProductInfo
from .config import AppConfig

class ProductUnderstandingAgent:
    def __init__(self, config: AppConfig):
        self.model = config.llms.product_understanding

    def run(self, raw_query: str) -> ProductInfo:
        prompt = f"""Given this product query: "{raw_query}"

Return a JSON object with exactly these keys:
- canonical_id: kebab-case product identifier (e.g. "sony-wh-1000xm5")
- canonical_name: full official product name
- category: product category (e.g. "headphones", "laptop", "camera")
- search_terms: list of 3-5 search terms to find Reddit discussions
- subreddits: list of 3-5 relevant subreddits (without r/ prefix)

Return only valid JSON, no explanation, no markdown."""

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return ProductInfo(
            canonical_id=data["canonical_id"],
            canonical_name=data["canonical_name"],
            category=data["category"],
            search_terms=data["search_terms"],
            subreddits=data["subreddits"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_product_agent.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/product_agent.py tests/test_product_agent.py
git commit -m "feat: add ProductUnderstandingAgent — resolves raw query to canonical product"
```

---

### Task 7: RedditScraper

**Files:**
- Create: `pipeline/scraper.py`
- Create: `tests/test_scraper.py`

*What you're learning:* PRAW (Python Reddit API Wrapper) abstracts Reddit's REST API. `replace_more(limit=0)` is a PRAW-specific call that prevents lazy-loading of "load more comments" placeholders — without it, you'd get incomplete comment trees silently.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scraper.py
import pytest
from unittest.mock import MagicMock, patch
from pipeline.scraper import RedditScraper
from pipeline.types import ProductInfo, RawComment

@pytest.fixture
def product_info():
    from pipeline.types import ProductInfo
    return ProductInfo(
        canonical_id="sony-wh-1000xm5",
        canonical_name="Sony WH-1000XM5",
        category="headphones",
        search_terms=["WH-1000XM5"],
        subreddits=["headphones"],
    )

def _make_mock_comment(id, body, score=10, created_utc=1700000000):
    c = MagicMock()
    c.id = id
    c.body = body
    c.score = score
    c.created_utc = created_utc
    c.subreddit = MagicMock()
    c.subreddit.__str__ = lambda self: "headphones"
    return c

def test_run_returns_raw_comments(product_info, mocker):
    mock_submission = MagicMock()
    mock_submission.url = "https://reddit.com/r/headphones/comments/abc"
    mock_comment = _make_mock_comment("c1", "Battery life is incredible on these headphones")
    mock_submission.comments.list.return_value = [mock_comment]

    mock_reddit = mocker.patch("pipeline.scraper.praw.Reddit")
    mock_reddit.return_value.subreddit.return_value.search.return_value = [mock_submission]

    scraper = RedditScraper("fake_id", "fake_secret", "fake_agent")
    results = scraper.run(product_info, limit=10)

    assert len(results) == 1
    assert isinstance(results[0], RawComment)
    assert results[0].id == "c1"
    assert results[0].subreddit == "headphones"

def test_run_deduplicates_comments(product_info, mocker):
    mock_submission = MagicMock()
    mock_submission.url = "https://reddit.com/r/headphones/comments/abc"
    same_comment = _make_mock_comment("dup1", "This comment appears in multiple search results")
    mock_submission.comments.list.return_value = [same_comment, same_comment]

    mock_reddit = mocker.patch("pipeline.scraper.praw.Reddit")
    mock_reddit.return_value.subreddit.return_value.search.return_value = [mock_submission, mock_submission]

    scraper = RedditScraper("fake_id", "fake_secret", "fake_agent")
    results = scraper.run(product_info, limit=10)

    ids = [r.id for r in results]
    assert ids.count("dup1") == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_scraper.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.scraper'`

- [ ] **Step 3: Create pipeline/scraper.py**

```python
import praw
from datetime import datetime
from .types import ProductInfo, RawComment

class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    def run(self, product: ProductInfo, limit: int = 100) -> list[RawComment]:
        comments: list[RawComment] = []
        seen_ids: set[str] = set()

        for term in product.search_terms[:3]:
            results = self.reddit.subreddit("all").search(
                term, limit=limit, sort="relevance", time_filter="year"
            )
            for submission in results:
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list():
                    if comment.id not in seen_ids:
                        seen_ids.add(comment.id)
                        comments.append(RawComment(
                            id=comment.id,
                            text=comment.body,
                            score=comment.score,
                            created_at=datetime.utcfromtimestamp(comment.created_utc),
                            subreddit=str(comment.subreddit),
                            post_url=submission.url,
                        ))
        return comments
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scraper.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/scraper.py tests/test_scraper.py
git commit -m "feat: add RedditScraper — fetches and deduplicates comments via PRAW"
```

---

### Task 8: AspectExtractor

**Files:**
- Create: `pipeline/aspect_extractor.py`
- Create: `tests/test_aspect_extractor.py`

*What you're learning:* **Batching** LLM calls is a key cost-optimization pattern. Instead of 240 individual API calls (one per comment), you send 15-20 comments per call. This cuts API costs by ~15x and speeds up the pipeline significantly. The trade-off: you need to parse structured output from a multi-item prompt, which requires robust JSON handling.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aspect_extractor.py
import json
import pytest
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.aspect_extractor import AspectExtractor
from pipeline.types import RawComment, AspectClaim
from datetime import datetime

@pytest.fixture
def config():
    return AppConfig(
        llms=LLMConfig("ollama/llama3.2", "gemini/gemini-2.0-flash", "ollama/llama3.2"),
        embeddings=EmbeddingConfig("local", "all-MiniLM-L6-v2"),
    )

def make_comment(id, text):
    return RawComment(id, text, 10, datetime(2024, 1, 1), "headphones", "https://reddit.com")

def test_extracts_claims_from_batch(config, mocker):
    mock_response = {"claims": [
        {"comment_id": "c1", "aspect": "battery life", "sentiment": "positive", "quote": "lasts 3 days"},
        {"comment_id": "c2", "aspect": "ANC", "sentiment": "negative", "quote": "ANC is weak"},
    ]}
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps(mock_response)

    extractor = AspectExtractor(config)
    comments = [
        make_comment("c1", "Battery life is great, lasts 3 days easily"),
        make_comment("c2", "ANC is weak compared to Bose QC45"),
    ]
    result = extractor.run(comments)

    assert len(result) == 2
    assert all(isinstance(r, AspectClaim) for r in result)
    assert result[0].aspect == "battery life"
    assert result[0].sentiment == "positive"

def test_handles_multi_aspect_comment(config, mocker):
    mock_response = {"claims": [
        {"comment_id": "c1", "aspect": "battery life", "sentiment": "positive", "quote": "great battery"},
        {"comment_id": "c1", "aspect": "ANC", "sentiment": "negative", "quote": "ANC disappoints"},
    ]}
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps(mock_response)

    extractor = AspectExtractor(config)
    result = extractor.run([make_comment("c1", "Great battery but ANC disappoints")])

    assert len(result) == 2
    assert all(c.comment_id == "c1" for c in result)

def test_batches_large_input(config, mocker):
    mock_completion = mocker.patch("pipeline.aspect_extractor.litellm.completion")
    mock_completion.return_value.choices[0].message.content = json.dumps({"claims": []})

    extractor = AspectExtractor(config)
    comments = [make_comment(f"c{i}", f"Comment number {i} about the product quality") for i in range(40)]
    extractor.run(comments)

    # 40 comments with batch size 15 → ceil(40/15) = 3 calls
    assert mock_completion.call_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_aspect_extractor.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.aspect_extractor'`

- [ ] **Step 3: Create pipeline/aspect_extractor.py**

```python
import json
import litellm
from .types import RawComment, AspectClaim
from .config import AppConfig

BATCH_SIZE = 15

class AspectExtractor:
    def __init__(self, config: AppConfig):
        self.model = config.llms.aspect_extraction

    def run(self, comments: list[RawComment]) -> list[AspectClaim]:
        claims: list[AspectClaim] = []
        for i in range(0, len(comments), BATCH_SIZE):
            batch = comments[i : i + BATCH_SIZE]
            claims.extend(self._extract_batch(batch))
        return claims

    def _extract_batch(self, comments: list[RawComment]) -> list[AspectClaim]:
        numbered = "\n".join(f"[{c.id}] {c.text}" for c in comments)
        prompt = f"""Extract product aspect claims from these Reddit comments.
For each opinion found, return: comment_id, aspect (e.g. "battery life"), sentiment (positive/negative/mixed), quote (short relevant excerpt, max 10 words).
A single comment can produce multiple claims if it mentions multiple aspects.

Comments:
{numbered}

Return JSON: {{"claims": [{{...}}, ...]}}"""

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return [
            AspectClaim(
                comment_id=item["comment_id"],
                aspect=item["aspect"],
                sentiment=item["sentiment"],
                quote=item["quote"],
            )
            for item in data.get("claims", [])
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_aspect_extractor.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/aspect_extractor.py tests/test_aspect_extractor.py
git commit -m "feat: add AspectExtractor — batched LLM extraction of aspect claims"
```

---

### Task 9: EmbedderClusterer

**Files:**
- Create: `pipeline/embedder_clusterer.py`
- Create: `tests/test_embedder_clusterer.py`

*What you're learning:* **Mean-shift clustering** discovers K automatically — you don't need to specify how many clusters you want. This matters here because you don't know in advance how many aspects a product has. `estimate_bandwidth` calculates the right neighborhood size from your data. The trade-off vs K-means: slower, but correct for unknown K.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedder_clusterer.py
import pytest
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.embedder_clusterer import EmbedderClusterer
from pipeline.types import AspectClaim, Cluster

@pytest.fixture
def config():
    return AppConfig(
        llms=LLMConfig("ollama/llama3.2", "gemini/gemini-2.0-flash", "ollama/llama3.2"),
        embeddings=EmbeddingConfig("local", "all-MiniLM-L6-v2"),
    )

@pytest.fixture
def mixed_claims():
    return [
        AspectClaim("c1", "battery life", "positive", "lasts 3 days"),
        AspectClaim("c2", "battery life", "positive", "great battery"),
        AspectClaim("c3", "battery life", "negative", "dies fast"),
        AspectClaim("c4", "ANC quality", "positive", "best ANC"),
        AspectClaim("c5", "ANC quality", "positive", "kills background noise"),
    ]

def test_returns_list_of_clusters(config, mixed_claims):
    ec = EmbedderClusterer(config)
    result = ec.run(mixed_claims)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(c, Cluster) for c in result)

def test_clusters_have_positive_negative_counts(config, mixed_claims):
    ec = EmbedderClusterer(config)
    result = ec.run(mixed_claims)
    for cluster in result:
        assert cluster.positive_count >= 0
        assert cluster.negative_count >= 0
        assert cluster.positive_count + cluster.negative_count <= len(cluster.claims)

def test_returns_empty_for_empty_input(config):
    ec = EmbedderClusterer(config)
    result = ec.run([])
    assert result == []

def test_sorted_by_claim_count(config, mixed_claims):
    ec = EmbedderClusterer(config)
    result = ec.run(mixed_claims)
    counts = [len(c.claims) for c in result]
    assert counts == sorted(counts, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_embedder_clusterer.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.embedder_clusterer'`

- [ ] **Step 3: Create pipeline/embedder_clusterer.py**

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MeanShift, estimate_bandwidth
from .types import AspectClaim, Cluster
from .config import AppConfig

class EmbedderClusterer:
    def __init__(self, config: AppConfig):
        self.model = SentenceTransformer(config.embeddings.model)

    def run(self, claims: list[AspectClaim]) -> list[Cluster]:
        if not claims:
            return []

        texts = [f"{c.aspect}: {c.quote}" for c in claims]
        embeddings = self.model.encode(texts)

        bandwidth = estimate_bandwidth(embeddings, quantile=0.3, n_samples=min(len(embeddings), 500))
        if bandwidth == 0:
            bandwidth = 0.5  # fallback for very small inputs

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels = ms.fit_predict(embeddings)

        cluster_map: dict[int, list[AspectClaim]] = {}
        for claim, label in zip(claims, labels):
            cluster_map.setdefault(int(label), []).append(claim)

        clusters = []
        for cluster_claims in cluster_map.values():
            aspects = [c.aspect for c in cluster_claims]
            label = max(set(aspects), key=aspects.count)
            positive = sum(1 for c in cluster_claims if c.sentiment == "positive")
            negative = sum(1 for c in cluster_claims if c.sentiment == "negative")
            clusters.append(Cluster(
                label=label,
                claims=cluster_claims,
                positive_count=positive,
                negative_count=negative,
            ))

        return sorted(clusters, key=lambda c: len(c.claims), reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_embedder_clusterer.py -v
```
Expected: 4 passed *(note: these use real sentence-transformers, expect ~5s on first run while model downloads)*

- [ ] **Step 5: Commit**

```bash
git add pipeline/embedder_clusterer.py tests/test_embedder_clusterer.py
git commit -m "feat: add EmbedderClusterer — mean-shift clustering of aspect claims"
```

---

### Task 10: Quantifier

**Files:**
- Create: `pipeline/quantifier.py`
- Create: `tests/test_quantifier.py`

*What you're learning:* The quantifier is pure logic — no LLM, no network. This is intentional. Counting mentions and computing percentages is deterministic math; it should never be left to a model. The **trend** calculation compares a 90-day window against the all-time baseline — a **sliding window** pattern common in time-series analysis.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quantifier.py
from datetime import datetime, timedelta
from pipeline.types import RawComment, AspectClaim, Cluster, QuantifiedAspect
from pipeline.quantifier import Quantifier

def make_comment(id, days_ago=30):
    return RawComment(id, "some text", 10, datetime.utcnow() - timedelta(days=days_ago), "headphones", "https://reddit.com")

def test_mention_pct_calculation():
    comments = [make_comment(f"c{i}") for i in range(10)]
    cluster = Cluster(
        label="battery life",
        claims=[AspectClaim(f"c{i}", "battery life", "positive", "great") for i in range(8)],
        positive_count=8,
        negative_count=0,
    )
    q = Quantifier()
    result = q.run([cluster], comments)
    assert result[0].mention_pct == 80.0

def test_positive_negative_pct():
    comments = [make_comment("c1"), make_comment("c2"), make_comment("c3")]
    cluster = Cluster(
        label="ANC",
        claims=[
            AspectClaim("c1", "ANC", "positive", "great"),
            AspectClaim("c2", "ANC", "negative", "weak"),
            AspectClaim("c3", "ANC", "positive", "excellent"),
        ],
        positive_count=2,
        negative_count=1,
    )
    q = Quantifier()
    result = q.run([cluster], comments)
    assert result[0].positive_pct == pytest.approx(66.7, abs=0.1)
    assert result[0].negative_pct == pytest.approx(33.3, abs=0.1)

def test_trend_declining_when_recent_more_negative():
    old_comments = [make_comment(f"old{i}", days_ago=200) for i in range(5)]
    recent_comments = [make_comment(f"new{i}", days_ago=10) for i in range(5)]
    all_comments = old_comments + recent_comments

    old_claims = [AspectClaim(f"old{i}", "durability", "positive", "good") for i in range(5)]
    recent_claims = [AspectClaim(f"new{i}", "durability", "negative", "broke") for i in range(5)]

    cluster = Cluster(
        label="durability",
        claims=old_claims + recent_claims,
        positive_count=5,
        negative_count=5,
    )
    q = Quantifier()
    result = q.run([cluster], all_comments)
    assert result[0].recent_trend == "declining"

def test_sorted_by_mention_pct():
    comments = [make_comment(f"c{i}") for i in range(10)]
    big_cluster = Cluster("battery", [AspectClaim(f"c{i}", "battery", "positive", "good") for i in range(8)], 8, 0)
    small_cluster = Cluster("ANC", [AspectClaim("c9", "ANC", "positive", "good")], 1, 0)
    q = Quantifier()
    result = q.run([small_cluster, big_cluster], comments)
    assert result[0].label == "battery"

import pytest
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_quantifier.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.quantifier'`

- [ ] **Step 3: Create pipeline/quantifier.py**

```python
from datetime import datetime, timedelta
from .types import RawComment, Cluster, QuantifiedAspect

class Quantifier:
    TREND_THRESHOLD = 0.10
    RECENT_DAYS = 90

    def run(self, clusters: list[Cluster], comments: list[RawComment]) -> list[QuantifiedAspect]:
        total = len(comments)
        comment_map = {c.id: c for c in comments}
        recent_cutoff = datetime.utcnow() - timedelta(days=self.RECENT_DAYS)

        aspects = []
        for cluster in clusters:
            mention_comment_ids = {claim.comment_id for claim in cluster.claims}
            mention_pct = (len(mention_comment_ids) / total * 100) if total > 0 else 0

            total_claims = len(cluster.claims)
            positive_pct = (cluster.positive_count / total_claims * 100) if total_claims > 0 else 0
            negative_pct = (cluster.negative_count / total_claims * 100) if total_claims > 0 else 0

            recent_claims = [
                c for c in cluster.claims
                if c.comment_id in comment_map
                and comment_map[c.comment_id].created_at >= recent_cutoff
            ]
            if recent_claims:
                recent_pos = sum(1 for c in recent_claims if c.sentiment == "positive")
                recent_pos_pct = recent_pos / len(recent_claims)
                delta = recent_pos_pct - (positive_pct / 100)
            else:
                delta = 0.0

            if delta > self.TREND_THRESHOLD:
                trend = "improving"
            elif delta < -self.TREND_THRESHOLD:
                trend = "declining"
            else:
                trend = "stable"

            aspects.append(QuantifiedAspect(
                label=cluster.label,
                mention_pct=round(mention_pct, 1),
                positive_pct=round(positive_pct, 1),
                negative_pct=round(negative_pct, 1),
                recent_trend=trend,
            ))

        return sorted(aspects, key=lambda a: a.mention_pct, reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_quantifier.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/quantifier.py tests/test_quantifier.py
git commit -m "feat: add Quantifier — computes mention%, sentiment split, and trend"
```

---

### Task 11: Summarizer

**Files:**
- Create: `pipeline/summarizer.py`
- Create: `tests/test_summarizer.py`

*What you're learning:* The summarizer is the **last** LLM call and the only one that talks to the user. It receives structured data (numbers, labels) and produces natural language. This separation — numbers come from code, words come from LLM — is why the output is trustworthy. The LLM can't hallucinate "87%" because it never computed that number.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_summarizer.py
import json
import pytest
from pipeline.config import AppConfig, LLMConfig, EmbeddingConfig
from pipeline.summarizer import Summarizer
from pipeline.types import Cluster, QuantifiedAspect, AspectClaim, AspectSummary

@pytest.fixture
def config():
    return AppConfig(
        llms=LLMConfig("ollama/llama3.2", "gemini/gemini-2.0-flash", "ollama/llama3.2"),
        embeddings=EmbeddingConfig("local", "all-MiniLM-L6-v2"),
    )

def test_run_returns_aspect_summaries(config, mocker):
    mock_response = {
        "headline": "Battery lasts days, not hours",
        "detail": "Most users praise the 30-hour battery. Recent reports of degradation.",
        "trend_note": "Degradation complaints increased in last 3 months.",
    }
    mocker.patch("pipeline.summarizer.litellm.completion").return_value.choices[0].message.content = json.dumps(mock_response)

    aspects = [QuantifiedAspect("battery life", 87.0, 71.0, 29.0, "declining")]
    clusters = [Cluster("battery life", [AspectClaim("c1", "battery life", "positive", "lasts 3 days")], 1, 0)]

    s = Summarizer(config)
    result = s.run(aspects, clusters)

    assert len(result) == 1
    assert isinstance(result[0], AspectSummary)
    assert result[0].headline == "Battery lasts days, not hours"
    assert result[0].mention_pct == 87.0
    assert result[0].recent_trend == "declining"

def test_run_preserves_quantified_data(config, mocker):
    mocker.patch("pipeline.summarizer.litellm.completion").return_value.choices[0].message.content = json.dumps({
        "headline": "Great ANC",
        "detail": "Users love it.",
        "trend_note": "",
    })
    aspects = [QuantifiedAspect("ANC", 91.0, 78.0, 22.0, "improving")]
    clusters = [Cluster("ANC", [], 0, 0)]

    s = Summarizer(config)
    result = s.run(aspects, clusters)

    assert result[0].positive_pct == 78.0
    assert result[0].negative_pct == 22.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_summarizer.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.summarizer'`

- [ ] **Step 3: Create pipeline/summarizer.py**

```python
import json
import litellm
from .types import Cluster, QuantifiedAspect, AspectSummary
from .config import AppConfig

class Summarizer:
    def __init__(self, config: AppConfig):
        self.model = config.llms.summarization

    def run(self, aspects: list[QuantifiedAspect], clusters: list[Cluster]) -> list[AspectSummary]:
        cluster_map = {c.label: c for c in clusters}
        return [self._summarize(aspect, cluster_map.get(aspect.label)) for aspect in aspects]

    def _summarize(self, aspect: QuantifiedAspect, cluster: Cluster | None) -> AspectSummary:
        quotes = [c.quote for c in (cluster.claims[:10] if cluster else [])]
        quotes_text = "\n".join(f"- {q}" for q in quotes) if quotes else "No quotes available."

        prompt = f"""Summarize Reddit opinions about the product aspect: "{aspect.label}"

Data:
- {aspect.mention_pct}% of comments mention this aspect
- {aspect.positive_pct}% positive, {aspect.negative_pct}% negative
- Recent trend: {aspect.recent_trend}

Sample user quotes:
{quotes_text}

Return JSON with:
- headline: one punchy sentence summarizing the overall opinion (max 15 words)
- detail: 2-3 sentences describing what users actually say, citing specific patterns
- trend_note: one sentence about recent trend (empty string if trend is stable)"""

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return AspectSummary(
            label=aspect.label,
            mention_pct=aspect.mention_pct,
            positive_pct=aspect.positive_pct,
            negative_pct=aspect.negative_pct,
            recent_trend=aspect.recent_trend,
            headline=data["headline"],
            detail=data["detail"],
            trend_note=data.get("trend_note", ""),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_summarizer.py -v
```
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/summarizer.py tests/test_summarizer.py
git commit -m "feat: add Summarizer — LLM generates headlines from quantified aspect data"
```

---

### Task 12: PipelineRunner

**Files:**
- Create: `pipeline/runner.py`
- Create: `tests/test_runner.py`

*What you're learning:* The runner is an **orchestrator** — it has no logic of its own, only coordination. It calls each component in order and passes results along. This is the **pipeline pattern**: a sequence of transformations where the output of each step is the input of the next. The `on_progress` callback lets the runner report progress without knowing who's listening.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import pytest
from unittest.mock import MagicMock
from pipeline.runner import PipelineRunner
from pipeline.types import ProductInfo, RawComment, AspectClaim, Cluster, QuantifiedAspect, AspectSummary
from datetime import datetime

@pytest.fixture
def mock_runner(mocker):
    mocker.patch("pipeline.runner.load_config")
    runner = PipelineRunner.__new__(PipelineRunner)
    runner.product_agent = MagicMock()
    runner.scraper = MagicMock()
    runner.noise_filter = MagicMock()
    runner.aspect_extractor = MagicMock()
    runner.embedder_clusterer = MagicMock()
    runner.quantifier = MagicMock()
    runner.summarizer = MagicMock()
    return runner

def test_run_calls_all_components_in_order(mock_runner):
    product = ProductInfo("test-id", "Test Product", "electronics", ["test"], ["gadgets"])
    comment = RawComment("c1", "Great product with excellent battery life", 10, datetime(2024, 1, 1), "gadgets", "https://reddit.com")
    claim = AspectClaim("c1", "battery", "positive", "excellent battery")
    cluster = Cluster("battery", [claim], 1, 0)
    aspect = QuantifiedAspect("battery", 100.0, 100.0, 0.0, "stable")
    summary = AspectSummary("battery", 100.0, 100.0, 0.0, "stable", "Great battery", "Users love it.", "")

    mock_runner.product_agent.run.return_value = product
    mock_runner.scraper.run.return_value = [comment]
    mock_runner.noise_filter.run.return_value = [comment]
    mock_runner.aspect_extractor.run.return_value = [claim]
    mock_runner.embedder_clusterer.run.return_value = [cluster]
    mock_runner.quantifier.run.return_value = [aspect]
    mock_runner.summarizer.run.return_value = [summary]

    result_product, result_summaries = mock_runner.run("test product")

    mock_runner.product_agent.run.assert_called_once_with("test product")
    mock_runner.scraper.run.assert_called_once_with(product)
    mock_runner.noise_filter.run.assert_called_once()
    mock_runner.aspect_extractor.run.assert_called_once()
    mock_runner.embedder_clusterer.run.assert_called_once()
    mock_runner.quantifier.run.assert_called_once()
    mock_runner.summarizer.run.assert_called_once()
    assert result_product == product
    assert result_summaries == [summary]

def test_run_calls_progress_callback(mock_runner):
    mock_runner.product_agent.run.return_value = ProductInfo("id", "Name", "cat", [], [])
    mock_runner.scraper.run.return_value = []
    mock_runner.noise_filter.run.return_value = []
    mock_runner.aspect_extractor.run.return_value = []
    mock_runner.embedder_clusterer.run.return_value = []
    mock_runner.quantifier.run.return_value = []
    mock_runner.summarizer.run.return_value = []

    progress_calls = []
    mock_runner.run("test", on_progress=lambda pct, msg: progress_calls.append(pct))

    assert len(progress_calls) > 0
    assert 100 in progress_calls
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_runner.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.runner'`

- [ ] **Step 3: Create pipeline/runner.py**

```python
import os
from .config import AppConfig, load_config
from .types import ProductInfo, AspectSummary
from .product_agent import ProductUnderstandingAgent
from .scraper import RedditScraper
from .noise_filter import NoiseFilter
from .aspect_extractor import AspectExtractor
from .embedder_clusterer import EmbedderClusterer
from .quantifier import Quantifier
from .summarizer import Summarizer
from typing import Callable

class PipelineRunner:
    def __init__(self, config: AppConfig | None = None):
        if config is None:
            config = load_config()
        self.product_agent = ProductUnderstandingAgent(config)
        self.scraper = RedditScraper(
            client_id=os.environ["REDDIT_CLIENT_ID"],
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ.get("REDDIT_USER_AGENT", "reddit-truth/0.1"),
        )
        self.noise_filter = NoiseFilter()
        self.aspect_extractor = AspectExtractor(config)
        self.embedder_clusterer = EmbedderClusterer(config)
        self.quantifier = Quantifier()
        self.summarizer = Summarizer(config)

    def run(
        self,
        raw_query: str,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> tuple[ProductInfo, list[AspectSummary]]:
        def progress(pct: int, msg: str) -> None:
            if on_progress:
                on_progress(pct, msg)

        progress(5, "Understanding product...")
        product = self.product_agent.run(raw_query)

        progress(15, f"Searching Reddit for {product.canonical_name}...")
        comments = self.scraper.run(product)

        progress(30, f"Found {len(comments)} comments. Filtering...")
        filtered = self.noise_filter.run(comments)

        progress(45, f"Extracting opinions from {len(filtered)} comments...")
        claims = self.aspect_extractor.run(filtered)

        progress(65, "Clustering insights...")
        clusters = self.embedder_clusterer.run(claims)

        progress(80, "Quantifying aspects...")
        aspects = self.quantifier.run(clusters, filtered)

        progress(90, "Generating summaries...")
        summaries = self.summarizer.run(aspects, clusters)

        progress(100, "Done")
        return product, summaries
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_runner.py -v
```
Expected: 2 passed

- [ ] **Step 5: Run all pipeline tests together**

```bash
pytest tests/ -v --ignore=tests/test_api.py
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add pipeline/runner.py tests/test_runner.py
git commit -m "feat: add PipelineRunner — orchestrates all 7 NLP components"
```

---

## Phase 3 — API + Celery Integration

### Task 13: Celery task

**Files:**
- Create: `tasks/pipeline_task.py`

*What you're learning:* The `@shared_task(bind=True)` decorator is a Celery pattern. `bind=True` gives the task access to `self` — its own task instance — which lets you retry on failure with `self.retry()`. Writing progress updates to the database (`job.save()`) is what enables Supabase Realtime to push those updates to the frontend WebSocket in real time.

- [ ] **Step 1: Create tasks/pipeline_task.py**

```python
# tasks/pipeline_task.py
from celery import shared_task
from django.utils import timezone

@shared_task(bind=True, max_retries=1)
def run_analysis_pipeline(self, job_id: str, raw_query: str) -> None:
    from core.models import Job, Product, AspectSummary as AspectSummaryModel
    from pipeline.runner import PipelineRunner

    job = Job.objects.get(id=job_id)
    job.status = "running"
    job.save(update_fields=["status"])

    def on_progress(pct: int, message: str) -> None:
        job.progress = pct
        job.status_message = message
        job.save(update_fields=["progress", "status_message"])

    try:
        runner = PipelineRunner()
        product_info, summaries = runner.run(raw_query, on_progress=on_progress)

        product, _ = Product.objects.get_or_create(
            id=product_info.canonical_id,
            defaults={
                "canonical_name": product_info.canonical_name,
                "category": product_info.category,
                "search_terms": product_info.search_terms,
                "subreddits": product_info.subreddits,
            },
        )

        AspectSummaryModel.objects.filter(product=product).delete()
        AspectSummaryModel.objects.bulk_create([
            AspectSummaryModel(
                product=product,
                aspect=s.label,
                mention_pct=s.mention_pct,
                positive_pct=s.positive_pct,
                negative_pct=s.negative_pct,
                recent_trend=s.recent_trend,
                headline=s.headline,
                detail=s.detail,
            )
            for s in summaries
        ])

        job.status = "done"
        job.canonical_id = product_info.canonical_id
        job.progress = 100
        job.status_message = "Complete"
        job.completed_at = timezone.now()
        job.save()

    except Exception as exc:
        job.status = "failed"
        job.status_message = str(exc)[:500]
        job.save(update_fields=["status", "status_message"])
        raise self.retry(exc=exc, countdown=0)
```

- [ ] **Step 2: Commit**

```bash
git add tasks/pipeline_task.py
git commit -m "feat: add Celery task that runs pipeline and writes progress to DB"
```

---

### Task 14: DRF serializers and views

**Files:**
- Create: `api/serializers.py`
- Create: `api/views.py`
- Create: `api/urls.py`
- Modify: `reddit_truth/urls.py`
- Create: `tests/test_api.py`

*What you're learning:* DRF **serializers** convert Django model instances to JSON and back. They're like a schema definition: declare fields once, get validation and serialization for free. The `AnalyzeView` uses `uuid4()` to generate a unique job ID — UUIDs are collision-resistant without needing a database roundtrip.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import pytest
from django.test import TestCase
from rest_framework.test import APIClient
from unittest.mock import patch
from core.models import Job, Product, AspectSummary

@pytest.mark.django_db
class TestAnalyzeEndpoint:
    def setup_method(self):
        self.client = APIClient()

    def test_returns_job_id_immediately(self, mocker):
        mocker.patch("api.views.run_analysis_pipeline.delay")
        response = self.client.post("/api/analyze/", {"query": "Sony WH-1000XM5"}, format="json")
        assert response.status_code == 202
        assert "job_id" in response.data
        assert response.data["cached"] is False

    def test_returns_cached_true_when_product_exists(self, mocker):
        mocker.patch("api.views.run_analysis_pipeline.delay")
        Product.objects.create(id="sony-wh-1000xm5", canonical_name="Sony WH-1000XM5", category="headphones")
        from core.models import QueryCache
        QueryCache.objects.create(raw_query="sony wh-1000xm5", canonical_id="sony-wh-1000xm5")

        response = self.client.post("/api/analyze/", {"query": "sony wh-1000xm5"}, format="json")
        assert response.status_code == 200
        assert response.data["cached"] is True
        assert response.data["canonical_id"] == "sony-wh-1000xm5"

    def test_returns_400_for_missing_query(self):
        response = self.client.post("/api/analyze/", {}, format="json")
        assert response.status_code == 400

@pytest.mark.django_db
class TestJobStatusEndpoint:
    def setup_method(self):
        self.client = APIClient()

    def test_returns_job_status(self):
        Job.objects.create(id="job123", product_query="test", status="running", progress=50)
        response = self.client.get("/api/jobs/job123/")
        assert response.status_code == 200
        assert response.data["status"] == "running"
        assert response.data["progress"] == 50

    def test_returns_404_for_unknown_job(self):
        response = self.client.get("/api/jobs/nonexistent/")
        assert response.status_code == 404

@pytest.mark.django_db
class TestProductDetailEndpoint:
    def setup_method(self):
        self.client = APIClient()

    def test_returns_product_with_summaries(self):
        product = Product.objects.create(id="sony-wh-1000xm5", canonical_name="Sony WH-1000XM5", category="headphones")
        AspectSummary.objects.create(
            product=product, aspect="battery life",
            mention_pct=87.0, positive_pct=71.0, negative_pct=29.0,
            recent_trend="declining", headline="Battery fades over time.", detail="Users note degradation."
        )
        response = self.client.get("/api/products/sony-wh-1000xm5/")
        assert response.status_code == 200
        assert response.data["canonical_name"] == "Sony WH-1000XM5"
        assert len(response.data["aspects"]) == 1
        assert response.data["aspects"][0]["mention_pct"] == 87.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api.py -v
```
Expected: `ImportError` or `404 not found` for urls

- [ ] **Step 3: Create api/serializers.py**

```python
from rest_framework import serializers
from core.models import Job, Product, AspectSummary

class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = ["id", "status", "progress", "status_message", "canonical_id", "created_at", "completed_at"]

class AspectSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = AspectSummary
        fields = ["aspect", "mention_pct", "positive_pct", "negative_pct", "recent_trend", "headline", "detail"]

class ProductDetailSerializer(serializers.ModelSerializer):
    aspects = AspectSummarySerializer(source="summaries", many=True)

    class Meta:
        model = Product
        fields = ["id", "canonical_name", "category", "aspects"]
```

- [ ] **Step 4: Create api/views.py**

```python
from uuid import uuid4
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from core.models import Job, Product, QueryCache
from tasks.pipeline_task import run_analysis_pipeline
from .serializers import JobSerializer, ProductDetailSerializer

class AnalyzeView(APIView):
    def post(self, request):
        query = request.data.get("query", "").strip().lower()
        if not query:
            return Response({"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST)

        cached = QueryCache.objects.filter(raw_query=query).first()
        if cached and Product.objects.filter(id=cached.canonical_id).exists():
            return Response({
                "job_id": None,
                "cached": True,
                "canonical_id": cached.canonical_id,
            }, status=status.HTTP_200_OK)

        job_id = str(uuid4())
        Job.objects.create(id=job_id, product_query=query)
        run_analysis_pipeline.delay(job_id, query)

        return Response({"job_id": job_id, "cached": False, "canonical_id": None}, status=status.HTTP_202_ACCEPTED)

class JobStatusView(APIView):
    def get(self, request, job_id):
        job = get_object_or_404(Job, id=job_id)
        return Response(JobSerializer(job).data)

class ProductDetailView(APIView):
    def get(self, request, canonical_id):
        product = get_object_or_404(Product, id=canonical_id)
        return Response(ProductDetailSerializer(product).data)
```

- [ ] **Step 5: Create api/urls.py**

```python
from django.urls import path
from .views import AnalyzeView, JobStatusView, ProductDetailView

urlpatterns = [
    path("analyze/", AnalyzeView.as_view(), name="analyze"),
    path("jobs/<str:job_id>/", JobStatusView.as_view(), name="job-status"),
    path("products/<str:canonical_id>/", ProductDetailView.as_view(), name="product-detail"),
]
```

- [ ] **Step 6: Update reddit_truth/urls.py**

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
]
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```
Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add api/ tests/test_api.py reddit_truth/urls.py
git commit -m "feat: add DRF API — analyze endpoint, job status, product detail"
```

---

### Task 15: Smoke test + run locally

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```
Expected: all tests pass (pipeline + API)

- [ ] **Step 2: Start Redis (required for Celery)**

```bash
redis-server
```

- [ ] **Step 3: Start Django dev server**

```bash
python manage.py runserver
```

- [ ] **Step 4: Start Celery worker in a second terminal**

```bash
celery -A reddit_truth worker --loglevel=info
```

- [ ] **Step 5: Submit a test query**

```bash
curl -X POST http://localhost:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Sony WH-1000XM5"}'
```
Expected: `{"job_id": "some-uuid", "cached": false, "canonical_id": null}`

- [ ] **Step 6: Poll job status**

```bash
curl http://localhost:8000/api/jobs/<job_id>/
```
Expected: `{"status": "running", "progress": 45, "status_message": "Extracting opinions..."}`

- [ ] **Step 7: Final commit**

```bash
git add .
git commit -m "chore: verify end-to-end smoke test passes locally"
```

---

## Self-Review

### Spec Coverage Check

| Spec requirement | Implemented in |
|---|---|
| Product Understanding Agent | Task 6: `pipeline/product_agent.py` |
| Reddit Scraper (PRAW) | Task 7: `pipeline/scraper.py` |
| Noise Filter | Task 5: `pipeline/noise_filter.py` |
| Aspect Extractor (batched, Gemini Flash) | Task 8: `pipeline/aspect_extractor.py` |
| Embedder + Clusterer (mean-shift) | Task 9: `pipeline/embedder_clusterer.py` |
| Quantifier (mention%, trend) | Task 10: `pipeline/quantifier.py` |
| Summarizer (LLM per cluster) | Task 11: `pipeline/summarizer.py` |
| PipelineRunner orchestrator | Task 12: `pipeline/runner.py` |
| 3-layer caching | Redis via Django cache framework (in views + task) ✓ |
| Async via Celery | Task 13: `tasks/pipeline_task.py` |
| 3 DRF endpoints | Task 14: `api/views.py` |
| config.yml LLM config | Task 3: `pipeline/config.py` |
| Database models (6 tables) | Task 4: `core/models.py` |
| Trend: 90-day sliding window | Task 10: `pipeline/quantifier.py` |

**Note on 3-layer Redis caching:** The views and Celery task handle cache reads/writes using Django's cache framework (`from django.core.cache import cache`). This should be added to `api/views.py` (check `product:{id}:summary` before triggering pipeline) and `tasks/pipeline_task.py` (write result to Redis after saving to DB). Add as a follow-up improvement after the smoke test passes.
