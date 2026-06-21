import pytest
from core.models import Job, Product, AspectSummary as AspectSummaryModel


# ── POST /api/jobs/ ───────────────────────────────────────────────────────────

@pytest.mark.django_db
def test_create_job_returns_202_and_job_id(client, mocker, django_capture_on_commit_callbacks):
    mock_delay = mocker.patch("api.views.run_pipeline_task.delay")

    with django_capture_on_commit_callbacks(execute=True):
        resp = client.post(
            "/api/jobs/", {"query": "Sony WH-1000XM5"}, content_type="application/json"
        )

    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    job = Job.objects.get(id=job_id)
    assert job.status == "pending"
    assert job.product_query == "Sony WH-1000XM5"
    # dispatch happens only AFTER the row is committed
    mock_delay.assert_called_once_with(job_id)


@pytest.mark.django_db
def test_job_id_is_unguessable_uuid(client, mocker, django_capture_on_commit_callbacks):
    mocker.patch("api.views.run_pipeline_task.delay")
    with django_capture_on_commit_callbacks(execute=True):
        resp = client.post("/api/jobs/", {"query": "X"}, content_type="application/json")
    job_id = resp.json()["job_id"]
    # 32-char hex uuid4, not a small sequential integer
    assert len(job_id) == 32
    assert not job_id.isdigit()


@pytest.mark.django_db
def test_dispatch_deferred_until_commit(client, mocker):
    """Without committing, the task must NOT have been dispatched yet."""
    mock_delay = mocker.patch("api.views.run_pipeline_task.delay")
    # no capture-on-commit wrapper → the test transaction never commits
    client.post("/api/jobs/", {"query": "Sony WH-1000XM5"}, content_type="application/json")
    mock_delay.assert_not_called()


@pytest.mark.django_db
def test_create_job_rejects_empty_query(client, mocker):
    mocker.patch("api.views.run_pipeline_task.delay")
    resp = client.post("/api/jobs/", {"query": ""}, content_type="application/json")
    assert resp.status_code == 400
    assert Job.objects.count() == 0


@pytest.mark.django_db
def test_create_job_rejects_missing_query(client, mocker):
    mocker.patch("api.views.run_pipeline_task.delay")
    resp = client.post("/api/jobs/", {}, content_type="application/json")
    assert resp.status_code == 400


# ── GET /api/jobs/<id>/ ───────────────────────────────────────────────────────

@pytest.mark.django_db
def test_get_job_status(client):
    Job.objects.create(
        id="abc123", product_query="Sony XM5", status="running",
        progress=40, status_message="scraping",
    )
    resp = client.get("/api/jobs/abc123/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["progress"] == 40
    assert data["status_message"] == "scraping"


@pytest.mark.django_db
def test_get_missing_job_returns_404(client):
    resp = client.get("/api/jobs/does-not-exist/")
    assert resp.status_code == 404


# ── GET /api/products/<id>/summaries/ ─────────────────────────────────────────

@pytest.mark.django_db
def test_get_product_summaries(client):
    product = Product.objects.create(
        id="sony-xm5", canonical_name="Sony WH-1000XM5", category="headphones",
    )
    AspectSummaryModel.objects.create(
        product=product, aspect="battery", mention_pct=80.0, positive_pct=75.0,
        negative_pct=25.0, recent_trend="improving", headline="Great battery",
        detail="Lasts days", trend_note="Up recently",
    )

    resp = client.get("/api/products/sony-xm5/summaries/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["aspect"] == "battery"
    assert data[0]["headline"] == "Great battery"
    assert data[0]["trend_note"] == "Up recently"


@pytest.mark.django_db
def test_get_summaries_for_unknown_product_is_empty(client):
    resp = client.get("/api/products/nope/summaries/")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.django_db
def test_list_products_returns_recent(client):
    Product.objects.create(id="a", canonical_name="A", category="x", comment_count=10)
    Product.objects.create(id="b", canonical_name="B", category="y", comment_count=20)
    resp = client.get("/api/products/")
    assert resp.status_code == 200
    data = resp.json()
    assert {p["id"] for p in data} == {"a", "b"}
    assert data[0]["comment_count"] in (10, 20)


@pytest.mark.django_db
def test_product_detail_returns_fields(client):
    Product.objects.create(id="sony-xm5", canonical_name="Sony WH-1000XM5",
                           category="headphones", subreddits=["headphones"], comment_count=60)
    resp = client.get("/api/products/sony-xm5/")
    assert resp.status_code == 200
    d = resp.json()
    assert d["canonical_name"] == "Sony WH-1000XM5"
    assert d["comment_count"] == 60
    assert d["subreddits"] == ["headphones"]


@pytest.mark.django_db
def test_product_detail_404_for_unknown(client):
    assert client.get("/api/products/nope/").status_code == 404
