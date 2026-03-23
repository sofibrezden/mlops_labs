import pytest
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TestQualityGate:

    @pytest.fixture
    def quality_thresholds(self):
        return {
            "rmse_test_max": 150.0,
            "mae_test_max": 100.0,
            "r2_test_min": 0.5,
        }

    @pytest.fixture
    def model_path(self):
        if os.path.exists("models/best_model.pkl"):
            return "models/best_model.pkl"
        elif os.path.exists("data/models/model.pkl"):
            return "data/models/model.pkl"
        else:
            pytest.skip("No model found for quality gate testing")

    @pytest.fixture
    def test_data(self):
        test_path = "data/prepared/test.csv"
        if not os.path.exists(test_path):
            pytest.skip("Test data not found")

        df = pd.read_csv(test_path)
        X_test = df.drop("count", axis=1)
        y_test = df["count"]
        return X_test, y_test

    @pytest.fixture
    def metrics_from_file(self):
        metrics_file = "metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                return json.load(f)
        return None

    @pytest.fixture
    def computed_metrics(self, model_path, test_data):
        model = joblib.load(model_path)
        X_test, y_test = test_data

        y_pred = model.predict(X_test)

        return {
            "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae_test": mean_absolute_error(y_test, y_pred),
            "r2_test": r2_score(y_test, y_pred),
        }

    def test_rmse_below_threshold(self, computed_metrics, quality_thresholds):
        rmse = computed_metrics["rmse_test"]
        threshold = quality_thresholds["rmse_test_max"]

        assert rmse <= threshold, (
            f"RMSE {rmse:.2f} exceeds threshold {threshold}. " f"Model performance is below acceptable quality."
        )

    def test_mae_below_threshold(self, computed_metrics, quality_thresholds):
        mae = computed_metrics["mae_test"]
        threshold = quality_thresholds["mae_test_max"]

        assert mae <= threshold, (
            f"MAE {mae:.2f} exceeds threshold {threshold}. " f"Model performance is below acceptable quality."
        )

    def test_r2_above_threshold(self, computed_metrics, quality_thresholds):
        r2 = computed_metrics["r2_test"]
        threshold = quality_thresholds["r2_test_min"]

        assert r2 >= threshold, (
            f"R2 score {r2:.4f} is below threshold {threshold}. " f"Model performance is below acceptable quality."
        )

    def test_no_overfitting(self, model_path):
        if not os.path.exists("data/prepared/train.csv"):
            pytest.skip("Train data not found")

        model = joblib.load(model_path)

        train_df = pd.read_csv("data/prepared/train.csv")
        test_df = pd.read_csv("data/prepared/test.csv")

        X_train = train_df.drop("count", axis=1)
        y_train = train_df["count"]

        X_test = test_df.drop("count", axis=1)
        y_test = test_df["count"]

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        overfitting_gap = r2_train - r2_test

        assert overfitting_gap < 0.3, (
            f"Potential overfitting detected: R2 train={r2_train:.4f}, "
            f"R2 test={r2_test:.4f}, gap={overfitting_gap:.4f}"
        )

    def test_predictions_reasonable_range(self, model_path, test_data):
        model = joblib.load(model_path)
        X_test, y_test = test_data

        y_pred = model.predict(X_test)

        assert (y_pred >= 0).all(), "Model produces negative predictions"

        assert (y_pred <= 1500).all(), f"Model produces unreasonably high predictions (max: {y_pred.max():.0f})"

    def test_model_consistency(self, model_path, test_data):
        model = joblib.load(model_path)
        X_test, y_test = test_data

        X_sample = X_test.head(10)

        pred1 = model.predict(X_sample)
        pred2 = model.predict(X_sample)

        np.testing.assert_array_almost_equal(pred1, pred2, err_msg="Model predictions are not consistent")


class TestQualityGateFromFile:

    @pytest.fixture
    def metrics_file(self):
        return "metrics.json"

    @pytest.fixture
    def quality_thresholds(self):
        return {
            "rmse_test_max": 150.0,
            "mae_test_max": 100.0,
            "r2_test_min": 0.5,
        }

    def test_metrics_file_quality_gate(self, metrics_file, quality_thresholds):
        if not os.path.exists(metrics_file):
            pytest.skip("Metrics file not found, using computed metrics instead")

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        if "rmse_test" in metrics:
            assert (
                metrics["rmse_test"] <= quality_thresholds["rmse_test_max"]
            ), f"RMSE {metrics['rmse_test']:.2f} exceeds threshold {quality_thresholds['rmse_test_max']}"

        if "mae_test" in metrics:
            assert (
                metrics["mae_test"] <= quality_thresholds["mae_test_max"]
            ), f"MAE {metrics['mae_test']:.2f} exceeds threshold {quality_thresholds['mae_test_max']}"

        if "r2_test" in metrics:
            assert (
                metrics["r2_test"] >= quality_thresholds["r2_test_min"]
            ), f"R2 {metrics['r2_test']:.4f} is below threshold {quality_thresholds['r2_test_min']}"
