"""Unit tests for lambdas/quality_data/lambda_function.py.

Tests the pure evaluation logic (evaluate_fact_checks, evaluate_freshness,
evaluate_station_checks). These functions don't hit AWS — they take parsed
Athena rows and return boolean results.
"""
import pytest


@pytest.fixture
def dq(modules):
    return modules["dq_check_lambda"]


class TestBuildFactQuery:
    def test_contains_sample_limit(self, dq):
        sql = dq.build_fact_query()
        assert f"LIMIT {dq.SAMPLE_ROWS}" in sql
        assert "aqi" in sql
        assert "fact_aqi" in sql


class TestEvaluateFactChecks:
    def test_all_pass_when_perfect(self, dq):
        metrics = {
            "total_rows": "10000",
            "null_aqi": "0",
            "null_measured_at": "0",
            "null_dominant_pollutant": "0",
            "null_queried_city": "0",
            "aqi_out_of_range": "0",
            "distinct_cities": "ha-noi,ho-chi-minh-city,da-nang,gia-lai,cao-bang",
            "invalid_sources": "",
        }
        checks = dq.evaluate_fact_checks(metrics)
        assert all(c["passed"] for c in checks)

    def test_row_count_fails_below_threshold(self, dq):
        metrics = {
            "total_rows": "1",
            "null_aqi": "0", "null_measured_at": "0",
            "null_dominant_pollutant": "0", "null_queried_city": "0",
            "aqi_out_of_range": "0",
            "distinct_cities": "ha-noi,ho-chi-minh-city,da-nang,gia-lai,cao-bang",
            "invalid_sources": "",
        }
        row_count_check = next(c for c in dq.evaluate_fact_checks(metrics) if c["check"] == "row_count")
        assert row_count_check["passed"] is False

    def test_aqi_range_fails_on_invalid(self, dq):
        metrics = {
            "total_rows": "100", "null_aqi": "0", "null_measured_at": "0",
            "null_dominant_pollutant": "0", "null_queried_city": "0",
            "aqi_out_of_range": "5",
            "distinct_cities": "ha-noi,ho-chi-minh-city,da-nang,gia-lai,cao-bang",
            "invalid_sources": "",
        }
        check = next(c for c in dq.evaluate_fact_checks(metrics) if c["check"] == "aqi_range")
        assert check["passed"] is False
        assert check["invalid_count"] == 5

    def test_city_coverage_detects_missing_city(self, dq):
        metrics = {
            "total_rows": "100", "null_aqi": "0", "null_measured_at": "0",
            "null_dominant_pollutant": "0", "null_queried_city": "0",
            "aqi_out_of_range": "0",
            "distinct_cities": "ha-noi,ho-chi-minh-city",
            "invalid_sources": "",
        }
        check = next(c for c in dq.evaluate_fact_checks(metrics) if c["check"] == "city_coverage")
        assert check["passed"] is False

    def test_null_pct_exceeds_threshold(self, dq):
        # 50% nulls in aqi → should fail (default threshold 5%)
        metrics = {
            "total_rows": "100", "null_aqi": "50",
            "null_measured_at": "0", "null_dominant_pollutant": "0",
            "null_queried_city": "0", "aqi_out_of_range": "0",
            "distinct_cities": "ha-noi,ho-chi-minh-city,da-nang,gia-lai,cao-bang",
            "invalid_sources": "",
        }
        aqi_null_check = next(
            c for c in dq.evaluate_fact_checks(metrics)
            if c["check"] == "null_pct" and c["column"] == "aqi"
        )
        assert aqi_null_check["passed"] is False
        assert aqi_null_check["value"] == 50.0


class TestEvaluateDimChecks:
    def test_passes_with_all_cities(self, dq):
        metrics = {
            "total_stations": "24",
            "station_cities": "ha-noi,ho-chi-minh-city,da-nang,gia-lai,cao-bang",
        }
        checks = dq.evaluate_dim_checks(metrics)
        assert all(c["passed"] for c in checks)

    def test_fails_with_zero_stations(self, dq):
        metrics = {"total_stations": "0", "station_cities": ""}
        checks = dq.evaluate_dim_checks(metrics)
        row_count = next(c for c in checks if c["check"] == "dim_station_row_count")
        assert row_count["passed"] is False

    def test_handles_missing_metrics(self, dq):
        checks = dq.evaluate_dim_checks(None)
        assert len(checks) == 1
        assert checks[0]["passed"] is False


class TestEvaluateFreshness:
    def test_passes_with_fresh_rows(self, dq):
        from datetime import datetime, timezone
        result = dq.evaluate_freshness({"fresh_rows": "100"}, datetime.now(timezone.utc))
        assert result["passed"] is True

    def test_fails_with_no_rows(self, dq):
        from datetime import datetime, timezone
        result = dq.evaluate_freshness({"fresh_rows": "0"}, datetime.now(timezone.utc))
        assert result["passed"] is False

    def test_fails_with_no_result(self, dq):
        from datetime import datetime, timezone
        result = dq.evaluate_freshness(None, datetime.now(timezone.utc))
        assert result["passed"] is False
