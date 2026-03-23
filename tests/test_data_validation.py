import pytest
import pandas as pd
import os


class TestDataValidation:

    @pytest.fixture
    def train_data_path(self):
        return "data/prepared/train.csv"

    @pytest.fixture
    def test_data_path(self):
        return "data/prepared/test.csv"

    def test_train_data_exists(self, train_data_path):
        assert os.path.exists(train_data_path), f"Train data not found at {train_data_path}"

    def test_test_data_exists(self, test_data_path):
        assert os.path.exists(test_data_path), f"Test data not found at {test_data_path}"

    def test_train_data_not_empty(self, train_data_path):
        df = pd.read_csv(train_data_path)
        assert len(df) > 0, "Train data is empty"

    def test_test_data_not_empty(self, test_data_path):
        df = pd.read_csv(test_data_path)
        assert len(df) > 0, "Test data is empty"

    def test_train_data_has_target(self, train_data_path):
        df = pd.read_csv(train_data_path)
        assert "count" in df.columns, "Target column 'count' not found in train data"

    def test_test_data_has_target(self, test_data_path):
        df = pd.read_csv(test_data_path)
        assert "count" in df.columns, "Target column 'count' not found in test data"

    def test_train_data_required_columns(self, train_data_path):
        df = pd.read_csv(train_data_path)
        required_columns = [
            "season",
            "holiday",
            "workingday",
            "weather",
            "temp",
            "atemp",
            "humidity",
            "windspeed",
            "hour",
            "day",
            "month",
            "weekday",
            "count",
        ]

        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' not found in train data"

    def test_test_data_required_columns(self, test_data_path):
        df = pd.read_csv(test_data_path)
        required_columns = [
            "season",
            "holiday",
            "workingday",
            "weather",
            "temp",
            "atemp",
            "humidity",
            "windspeed",
            "hour",
            "day",
            "month",
            "weekday",
            "count",
        ]

        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' not found in test data"

    def test_train_data_no_nulls(self, train_data_path):
        df = pd.read_csv(train_data_path)
        null_counts = df.isnull().sum()
        assert null_counts.sum() == 0, f"Train data contains null values: {null_counts[null_counts > 0]}"

    def test_test_data_no_nulls(self, test_data_path):
        df = pd.read_csv(test_data_path)
        null_counts = df.isnull().sum()
        assert null_counts.sum() == 0, f"Test data contains null values: {null_counts[null_counts > 0]}"

    def test_train_data_types(self, train_data_path):
        df = pd.read_csv(train_data_path)

        numeric_columns = [
            "season",
            "holiday",
            "workingday",
            "weather",
            "temp",
            "atemp",
            "humidity",
            "windspeed",
            "hour",
            "day",
            "month",
            "weekday",
            "count",
        ]

        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric"

    def test_test_data_types(self, test_data_path):
        df = pd.read_csv(test_data_path)

        numeric_columns = [
            "season",
            "holiday",
            "workingday",
            "weather",
            "temp",
            "atemp",
            "humidity",
            "windspeed",
            "hour",
            "day",
            "month",
            "weekday",
            "count",
        ]

        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric"

    def test_target_values_positive(self, train_data_path, test_data_path):
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)

        assert (train_df["count"] >= 0).all(), "Train data contains negative target values"
        assert (test_df["count"] >= 0).all(), "Test data contains negative target values"

    def test_data_split_ratio(self, train_data_path, test_data_path):
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)

        total = len(train_df) + len(test_df)
        test_ratio = len(test_df) / total

        assert 0.15 <= test_ratio <= 0.25, f"Test split ratio {test_ratio:.2f} is outside expected range [0.15, 0.25]"

    def test_feature_ranges(self, train_data_path):
        df = pd.read_csv(train_data_path)

        assert df["season"].between(1, 4).all(), "Season values out of range [1, 4]"
        assert df["weather"].between(1, 4).all(), "Weather values out of range [1, 4]"
        assert df["hour"].between(0, 23).all(), "Hour values out of range [0, 23]"
        assert df["month"].between(1, 12).all(), "Month values out of range [1, 12]"
        assert df["weekday"].between(0, 6).all(), "Weekday values out of range [0, 6]"
        assert df["humidity"].between(0, 100).all(), "Humidity values out of range [0, 100]"
