set -e

echo "================================"
echo "Running ML Pipeline Test Suite"
echo "================================"
echo ""

echo "1. Running smoke tests..."
pytest tests/test_smoke.py -v --tb=short
echo ""

echo "2. Running data validation tests..."
pytest tests/test_data_validation.py -v --tb=short
echo ""

echo "3. Running artifact validation tests..."
pytest tests/test_artifacts.py -v --tb=short
echo ""

echo "4. Running quality gate tests..."
pytest tests/test_quality_gate.py -v --tb=short
echo ""

echo "================================"
echo "All tests completed successfully!"
echo "================================"
