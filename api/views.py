from uuid import uuid4

from django.db import transaction
from rest_framework import status
from rest_framework.generics import RetrieveAPIView, ListAPIView
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import Job, AspectSummary
from tasks.pipeline_task import run_pipeline_task
from .serializers import JobCreateSerializer, JobSerializer, AspectSummarySerializer


class JobCreateView(APIView):
    """POST a query → create a pending Job and dispatch the pipeline."""

    def post(self, request):
        serializer = JobCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Random, unguessable id: there is no auth, so the URL is the capability.
        job = Job.objects.create(
            id=uuid4().hex,
            product_query=serializer.validated_data["query"],
            status="pending",
        )

        # Defer dispatch until the row is committed, so the worker can never
        # read a job that a rollback would erase (and tests stay correct under
        # CELERY_TASK_ALWAYS_EAGER).
        transaction.on_commit(lambda: run_pipeline_task.delay(job.id))

        return Response({"job_id": job.id, "status": job.status}, status=status.HTTP_202_ACCEPTED)


class JobDetailView(RetrieveAPIView):
    """GET job status/progress — polled by the frontend."""
    queryset = Job.objects.all()
    serializer_class = JobSerializer
    lookup_field = "id"


class ProductSummariesView(ListAPIView):
    """GET the finished aspect summaries for a product."""
    serializer_class = AspectSummarySerializer

    def get_queryset(self):
        return AspectSummary.objects.filter(product_id=self.kwargs["canonical_id"])
