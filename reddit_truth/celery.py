import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reddit_truth.settings")
app = Celery("reddit_truth")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
