"""Unit tests for lambdas/ml_inference/lambda_function.py.

Stubs boto3 clients rather than using moto to keep tests fast and focused
on the Lambda's business logic (not AWS SDK behaviour).
"""
import io
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def ml(modules, monkeypatch):
    mod = modules["ml_inference_lambda"]
    # Replace the module-level boto3 clients with mocks so tests don't hit AWS.
    monkeypatch.setattr(mod, "sm", MagicMock())
    monkeypatch.setattr(mod, "s3", MagicMock())
    monkeypatch.setattr(mod, "sns", MagicMock())
    return mod


class TestLatestApprovedPackage:
    def test_returns_first_when_multiple(self, ml):
        ml.sm.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": "arn:aws:sagemaker:::mp/v2"},
                {"ModelPackageArn": "arn:aws:sagemaker:::mp/v1"},
            ]
        }
        ml.sm.describe_model_package.return_value = {"ModelPackageArn": "arn:aws:sagemaker:::mp/v2"}
        result = ml.latest_approved_package("grp")
        assert result["ModelPackageArn"].endswith("v2")

    def test_raises_when_empty(self, ml):
        ml.sm.list_model_packages.return_value = {"ModelPackageSummaryList": []}
        with pytest.raises(RuntimeError, match="No Approved model packages"):
            ml.latest_approved_package("empty-group")


class TestEnsureModel:
    def test_reuses_existing_model(self, ml):
        ml.sm.describe_model.return_value = {"ModelName": "existing"}
        name = ml.ensure_model("existing", "arn:xxx")
        assert name == "existing"
        ml.sm.create_model.assert_not_called()

    def test_creates_when_missing(self, ml):
        from botocore.exceptions import ClientError
        ml.sm.describe_model.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "not found"}},
            "DescribeModel",
        )
        name = ml.ensure_model("new-model", "arn:aws:sagemaker:::mp/v1")
        assert name == "new-model"
        ml.sm.create_model.assert_called_once()


class TestReadCsvFromS3:
    def test_filters_and_concatenates(self, ml):
        ml.s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": [
                {"Key": "prefix/a.out"},
                {"Key": "prefix/b.csv"},
                {"Key": "prefix/skip.json"},
            ]}
        ]

        def get_object(Bucket, Key):
            body = {"prefix/a.out": "x,y\n1,2", "prefix/b.csv": "x,y\n3,4"}[Key]
            return {"Body": io.BytesIO(body.encode("utf-8"))}

        ml.s3.get_object.side_effect = get_object
        rows = ml.read_csv_from_s3("bucket", "prefix")
        assert rows == [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}]


class TestHandler:
    def test_unknown_mode_raises(self, ml):
        with pytest.raises(ValueError, match="Unknown mode"):
            ml.lambda_handler({"mode": "bogus"}, None)

    def test_trigger_mode_invokes_trigger_jobs(self, ml):
        with patch.object(ml, "trigger_jobs", return_value={"forecast_job": "fj"}) as tj:
            result = ml.lambda_handler(
                {"mode": "trigger", "run_id": "r1", "features_input_uri": "s3://x/"},
                None,
            )
            tj.assert_called_once_with("s3://x/", "r1")
            assert result["forecast_job"] == "fj"

    def test_collect_mode_invokes_collect(self, ml):
        jobs_info = {"run_id": "r1"}
        with patch.object(ml, "collect_and_write_gold", return_value={"ok": True}) as cg:
            result = ml.lambda_handler(
                {"mode": "collect", "jobs_info": jobs_info},
                None,
            )
            cg.assert_called_once_with(jobs_info)
            assert result["ok"] is True


class TestCollectAndWriteGold:
    @pytest.fixture
    def stub_jobs(self):
        return {
            "run_id": "r1",
            "forecast_output": "s3://bkt/forecast/",
            "anomaly_output":  "s3://bkt/anomaly/",
            "forecast_model_package": "arn:fc",
            "anomaly_model_package":  "arn:an",
        }

    def test_parses_csv_rows_and_writes_jsonl(self, ml, stub_jobs):
        forecast_rows = [{"predicted_aqi": "95.0"}, {"predicted_aqi": "88.2"}]
        anomaly_rows  = [{"anomaly_score": "0.2", "is_anomaly": "0"},
                         {"anomaly_score": "0.9", "is_anomaly": "1"}]

        def fake_read(bucket, prefix):
            return forecast_rows if "forecast" in prefix else anomaly_rows

        with patch.object(ml, "read_csv_from_s3", side_effect=fake_read), \
             patch.object(ml, "send_alert") as send_alert:
            result = ml.collect_and_write_gold(stub_jobs)
            send_alert.assert_called_once()  # High-severity anomaly triggers alert

        assert result["forecast_count"] == 2
        assert result["anomaly_count"] == 2
        assert result["high_severity_count"] == 1
        # 2 put_object calls: one for forecast, one for anomalies
        assert ml.s3.put_object.call_count == 2

    def test_skips_rows_with_invalid_predicted_aqi(self, ml, stub_jobs):
        forecast_rows = [{"predicted_aqi": "not-a-number"}, {"predicted_aqi": "80"}]
        with patch.object(ml, "read_csv_from_s3", side_effect=[forecast_rows, []]):
            result = ml.collect_and_write_gold(stub_jobs)
        assert result["forecast_count"] == 1
