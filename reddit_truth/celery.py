import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "reddit_truth.settings")
app = Celery("reddit_truth")
app.config_from_object("django.conf:settings", namespace="CELERY")
# Discover <app>/tasks.py in installed apps, plus our top-level tasks/ package
# whose modules don't follow the <app>.tasks naming convention.
app.autodiscover_tasks()
app.autodiscover_tasks(["tasks"], related_name="pipeline_task")
