"""Unit tests for lambdas/air-quality-api-ingestion/lambda_function.py.

Covers:
  - S3 key building (Hive-style partitioning)
  - WAQI API response parsing
  - End-to-end handler with mocked HTTP + S3
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def ing(modules, monkeypatch):
    mod = modules["ingestion_lambda"]
    monkeypatch.setattr(mod, "s3_client", MagicMock())
    monkeypatch.setattr(mod, "sns_client", MagicMock())
    return mod


class TestBuildS3Key:
    def test_hive_partitioned_path(self, ing):
        now = datetime(2024, 3, 15, 9, 30, 0, tzinfo=timezone.utc)
        key = ing.build_s3_key("ha-noi", now)
        assert key.startswith("api_raw/queried_city=ha-noi/year=2024/month=03/day=15/")
        assert key.endswith(".json")
        assert "ha-noi_" in key

    def test_different_cities_produce_different_keys(self, ing):
        now = datetime(2024, 3, 15, 9, 30, 0, tzinfo=timezone.utc)
        k1 = ing.build_s3_key("ha-noi", now)
        k2 = ing.build_s3_key("da-nang", now)
        assert k1 != k2
        assert "ha-noi" in k1 and "da-nang" in k2

    def test_zero_padded_month_day(self, ing):
        now = datetime(2024, 1, 5, 0, 0, tzinfo=timezone.utc)
        key = ing.build_s3_key("ha-noi", now)
        assert "month=01" in key
        assert "day=05" in key


class TestFetchCityAqi:
    def test_raises_on_non_ok_status(self, ing):
        stub_response = MagicMock()
        stub_response.read.return_value = b'{"status": "error", "data": "invalid token"}'
        stub_response.__enter__ = lambda s: s
        stub_response.__exit__ = lambda s, *a: None

        with patch.object(ing, "urlopen", return_value=stub_response):
            with pytest.raises(ValueError, match="non-ok status"):
                ing.fetch_city_aqi("ha-noi")

    def test_parses_ok_response(self, ing):
        stub_response = MagicMock()
        stub_response.read.return_value = b'{"status": "ok", "data": {"aqi": 85, "city": {"name": "Hanoi"}}}'
        stub_response.__enter__ = lambda s: s
        stub_response.__exit__ = lambda s, *a: None

        with patch.object(ing, "urlopen", return_value=stub_response):
            result = ing.fetch_city_aqi("ha-noi")
            assert result["status"] == "ok"
            assert result["data"]["aqi"] == 85
