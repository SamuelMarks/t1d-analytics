"""Tests for the models."""

from t1d_analytics.models import DatasetInfo


def test_dataset_info_creation() -> None:
    """Test that DatasetInfo can be instantiated correctly."""
    info = DatasetInfo(protocol="Test", dataset_url="http://a", document_url=None)
    assert info.protocol == "Test"
    assert info.dataset_url == "http://a"
    assert info.document_url is None
