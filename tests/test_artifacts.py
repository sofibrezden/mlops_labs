import pytest
import os
import json
import joblib


class TestArtifacts:

    @pytest.fixture
    def model_dir(self):
        return "models"

    @pytest.fixture
    def data_model_dir(self):
        return "data/models"

    def test_model_directory_exists(self, model_dir):
        assert os.path.exists(model_dir), f"Model directory not found at {model_dir}"

    def test_best_model_exists(self, model_dir):
        model_path = os.path.join(model_dir, "best_model.pkl")
        assert os.path.exists(model_path), f"Best model not found at {model_path}"

    def test_model_pkl_exists(self, data_model_dir):
        if os.path.exists(data_model_dir):
            model_path = os.path.join(data_model_dir, "model.pkl")
            assert os.path.exists(model_path), f"Model not found at {model_path}"

    def test_best_model_loadable(self, model_dir):
        model_path = os.path.join(model_dir, "best_model.pkl")
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                assert model is not None, "Loaded model is None"
            except Exception as e:
                pytest.fail(f"Failed to load model: {str(e)}")

    def test_model_has_predict_method(self, model_dir):
        model_path = os.path.join(model_dir, "best_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            assert hasattr(model, "predict"), "Model does not have predict method"
            assert callable(model.predict), "Model predict is not callable"

    def test_model_has_feature_importances(self, model_dir):
        model_path = os.path.join(model_dir, "best_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            assert hasattr(model, "feature_importances_"), "Model does not have feature_importances_"

    def test_model_file_size(self, model_dir):
        model_path = os.path.join(model_dir, "best_model.pkl")
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            assert file_size > 1000, f"Model file size {file_size} bytes is suspiciously small"
            assert file_size < 500 * 1024 * 1024, f"Model file size {file_size} bytes is suspiciously large (>500MB)"

    def test_feature_importance_plot_exists(self, data_model_dir):
        if os.path.exists(data_model_dir):
            plot_path = os.path.join(data_model_dir, "feature_importance.png")
            if os.path.exists(plot_path):
                file_size = os.path.getsize(plot_path)
                assert file_size > 0, "Feature importance plot is empty"


class TestMetrics:

    @pytest.fixture
    def metrics_file(self):
        return "metrics.json"

    def test_metrics_file_exists_or_in_mlflow(self, metrics_file):
        if not os.path.exists(metrics_file):
            pytest.skip("Metrics file not found, assuming metrics are in MLflow")

    def test_metrics_file_valid_json(self, metrics_file):
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                assert isinstance(metrics, dict), "Metrics should be a dictionary"
            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to parse metrics JSON: {str(e)}")

    def test_metrics_contain_required_fields(self, metrics_file):
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            required_metrics = ["rmse_test", "mae_test", "r2_test"]
            for metric in required_metrics:
                assert metric in metrics, f"Required metric '{metric}' not found in metrics file"

    def test_metrics_values_valid(self, metrics_file):
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            if "rmse_test" in metrics:
                assert metrics["rmse_test"] >= 0, "RMSE should be non-negative"

            if "mae_test" in metrics:
                assert metrics["mae_test"] >= 0, "MAE should be non-negative"

            if "r2_test" in metrics:
                assert metrics["r2_test"] <= 1, "R2 should be <= 1"