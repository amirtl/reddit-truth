# Test settings — overrides production settings for local test runs.
# Uses SQLite so tests need no Supabase credentials.
# Usage: DJANGO_SETTINGS_MODULE=reddit_truth.settings_test pytest

from .settings import *  # noqa: F401, F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Use a real in-memory cache so cache tests work without Redis
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

# Celery runs tasks synchronously in tests — no worker needed
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
