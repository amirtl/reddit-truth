from rest_framework import serializers
from core.models import Job, AspectSummary


class JobCreateSerializer(serializers.Serializer):
    """Validates the submit payload. Never trust the client — require a query."""
    query = serializers.CharField(max_length=500, allow_blank=False, trim_whitespace=True)


class JobSerializer(serializers.ModelSerializer):
    """Read view of a job — what the frontend polls to drive the progress bar."""
    class Meta:
        model = Job
        fields = [
            "id", "product_query", "canonical_id", "status",
            "progress", "status_message", "created_at", "completed_at",
        ]


class AspectSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = AspectSummary
        fields = [
            "aspect", "mention_pct", "positive_pct", "negative_pct",
            "recent_trend", "headline", "detail", "trend_note", "generated_at",
        ]
