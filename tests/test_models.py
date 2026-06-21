import pytest
from core.models import Product


@pytest.mark.django_db
def test_product_comment_count_defaults_to_zero():
    p = Product.objects.create(id="x", canonical_name="X", category="c")
    assert p.comment_count == 0
