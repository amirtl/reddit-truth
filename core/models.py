from django.db import models


class Product(models.Model):
    id = models.CharField(max_length=200, primary_key=True)
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
    SENTIMENT_CHOICES = [
        ("positive", "Positive"),
        ("negative", "Negative"),
        ("mixed", "Mixed"),
    ]
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="aspect_claims")
    comment = models.ForeignKey(RawComment, on_delete=models.CASCADE, related_name="claims")
    aspect = models.CharField(max_length=200)
    sentiment = models.CharField(max_length=20, choices=SENTIMENT_CHOICES)
    quote = models.TextField()


class AspectSummary(models.Model):
    TREND_CHOICES = [
        ("improving", "Improving"),
        ("declining", "Declining"),
        ("stable", "Stable"),
    ]
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="summaries")
    aspect = models.CharField(max_length=200)
    mention_pct = models.FloatField()
    positive_pct = models.FloatField()
    negative_pct = models.FloatField()
    recent_trend = models.CharField(max_length=20, choices=TREND_CHOICES)
    headline = models.TextField()
    detail = models.TextField()
    trend_note = models.TextField(default="", blank=True)
    generated_at = models.DateTimeField(auto_now_add=True)


class QueryCache(models.Model):
    raw_query = models.CharField(max_length=500, primary_key=True)
    canonical_id = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
