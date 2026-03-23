# Test Suite Documentation

This directory contains comprehensive tests for the ML pipeline.

## Test Categories

### 1. Smoke Tests (`test_smoke.py`)
Quick tests to verify basic functionality:
- Package imports
- Project structure
- Script syntax validation

**Run:** `pytest tests/test_smoke.py -v`

### 2. Data Validation Tests (`test_data_validation.py`)
Validates input data quality and structure:
- Data file existence
- Required columns presence
- Data types validation
- No null values
- Feature value ranges
- Train/test split ratio
- Target value validation

**Run:** `pytest tests/test_data_validation.py -v`

### 3. Artifact Tests (`test_artifacts.py`)
Verifies model artifacts are created correctly:
- Model file existence
- Model loadability
- Model has required methods
- Model file size validation
- Metrics file validation
- Feature importance plots

**Run:** `pytest tests/test_artifacts.py -v`

### 4. Quality Gate Tests (`test_quality_gate.py`)
Ensures model meets quality thresholds:
- RMSE below threshold (≤ 150.0)
- MAE below threshold (≤ 100.0)
- R² above threshold (≥ 0.5)
- No overfitting detection
- Prediction range validation
- Model consistency

**Run:** `pytest tests/test_quality_gate.py -v`

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test category
```bash
pytest tests/test_smoke.py -v
pytest tests/test_data_validation.py -v
pytest tests/test_artifacts.py -v
pytest tests/test_quality_gate.py -v
```

### Run with coverage
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Run only smoke tests (fast)
```bash
pytest tests/test_smoke.py -v
```

## Quality Thresholds

The following thresholds are configured in `test_quality_gate.py`:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| RMSE   | ≤ 150.0   | Root Mean Squared Error on test set |
| MAE    | ≤ 100.0   | Mean Absolute Error on test set |
| R²     | ≥ 0.5     | R-squared score on test set |

## CI/CD Integration

These tests are automatically run in the GitHub Actions workflow:

1. **Pre-training tests**: Smoke tests and data validation
2. **Post-training tests**: Artifact validation and quality gates

The workflow fails if any quality gate test fails, preventing poor models from being deployed.

## Customizing Thresholds

To adjust quality thresholds, modify the `quality_thresholds` fixture in `tests/test_quality_gate.py`:

```python
@pytest.fixture
def quality_thresholds(self):
    return {
        "rmse_test_max": 150.0,  # Adjust as needed
        "mae_test_max": 100.0,   # Adjust as needed
        "r2_test_min": 0.5,      # Adjust as needed
    }
```
