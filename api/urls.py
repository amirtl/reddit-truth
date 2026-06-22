from django.urls import path
from .views import (
    JobCreateView, JobDetailView, ProductSummariesView,
    ProductListView, ProductDetailView,
)

urlpatterns = [
    path("jobs/", JobCreateView.as_view(), name="job-create"),
    path("jobs/<str:id>/", JobDetailView.as_view(), name="job-detail"),
    path("products/", ProductListView.as_view(), name="product-list"),
    path("products/<str:canonical_id>/", ProductDetailView.as_view(), name="product-detail"),
    path("products/<str:canonical_id>/summaries/", ProductSummariesView.as_view(), name="product-summaries"),
]
