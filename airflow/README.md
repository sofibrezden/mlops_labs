# Airflow Setup Guide

## Quick Start

### 1. Initialize Airflow

```bash
# Set Airflow UID (first time only)
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Start all services
docker-compose up -d
```

### 2. Access Airflow UI

- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### 3. Verify DAG

1. Navigate to the Airflow UI
2. Find `ml_training_pipeline` in the DAGs list
3. Toggle the DAG to "On"
4. Click "Trigger DAG" to run manually

## Directory Structure

```
airflow/
├── dags/                      # DAG definitions
│   └── ml_training_pipeline.py
├── logs/                      # Task execution logs
├── plugins/                   # Custom Airflow plugins
└── config/                    # Airflow configuration files
```

## DAG Overview: `ml_training_pipeline`

### Tasks Flow

```
check_data_availability
        ↓
check_dvc_updates
        ↓
prepare_data
        ↓
train_model
        ↓
evaluate_and_branch
    ↙       ↘
register_model   notify_failure
    ↓               ↓
    └─→ notify_success ←┘
```

### Task Descriptions

1. **check_data_availability**: Waits for raw data file
2. **check_dvc_updates**: Verifies DVC status
3. **prepare_data**: Runs data preprocessing
4. **train_model**: Trains ML model with MLflow tracking
5. **evaluate_and_branch**: Evaluates metrics and decides next step
6. **register_model**: Registers model in MLflow (if quality threshold met)
7. **notify_success/failure**: Terminal nodes

## Configuration

### Quality Thresholds

Edit `dags/ml_training_pipeline.py`:

```python
r2_threshold = 0.5      # Minimum R² score
rmse_threshold = 100    # Maximum RMSE
```

### Schedule

Default: `@daily` (runs once per day)

To change:
```python
schedule_interval='@hourly'  # or '0 0 * * *' for cron
```

## Monitoring

### View Logs

```bash
# Scheduler logs
docker-compose logs -f airflow-scheduler

# Webserver logs
docker-compose logs -f airflow-webserver
```

### Check DAG Status

```bash
# List all DAGs
docker-compose exec airflow-scheduler airflow dags list

# Check for import errors
docker-compose exec airflow-scheduler airflow dags list-import-errors

# Test specific task
docker-compose exec airflow-scheduler \
  airflow tasks test ml_training_pipeline check_data_availability 2024-01-01
```

## Troubleshooting

### DAG not appearing

```bash
# Check import errors
docker-compose exec airflow-scheduler airflow dags list-import-errors

# Restart scheduler
docker-compose restart airflow-scheduler
```

### Permission issues

```bash
# Fix ownership
sudo chown -R $USER:$USER airflow/

# Or set correct UID
echo "AIRFLOW_UID=$(id -u)" >> .env
docker-compose down -v
docker-compose up -d
```

### Database issues

```bash
# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

## Useful Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View running containers
docker-compose ps

# Enter scheduler container
docker-compose exec airflow-scheduler bash

# Trigger DAG manually
docker-compose exec airflow-scheduler \
  airflow dags trigger ml_training_pipeline

# Pause DAG
docker-compose exec airflow-scheduler \
  airflow dags pause ml_training_pipeline

# Unpause DAG
docker-compose exec airflow-scheduler \
  airflow dags unpause ml_training_pipeline
```

## Adding New DAGs

1. Create new Python file in `airflow/dags/`
2. Define your DAG using Airflow operators
3. Save the file
4. Wait ~30 seconds for Airflow to detect it
5. Refresh the UI to see your new DAG

## Best Practices

1. **Idempotency**: Ensure tasks can be re-run safely
2. **Logging**: Add informative logs to tasks
3. **Error Handling**: Use `on_failure_callback` for alerts
4. **Testing**: Test DAGs locally before deploying
5. **Documentation**: Add docstrings to DAG and tasks
