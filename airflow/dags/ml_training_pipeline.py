"""
ML Training Pipeline DAG for Apache Airflow
This DAG orchestrates the complete ML workflow including:
- Data availability check
- Data preparation
- Model training
- Model evaluation with branching logic
- Model registration in MLflow
"""

import os
import json
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Complete ML training pipeline with DVC and MLflow',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training', 'dvc', 'mlflow'],
)


def check_data_file_exists(**context):
    """Check if the raw data file exists"""
    data_file = '/opt/airflow/data/raw/dataset.csv'
    
    if os.path.exists(data_file):
        file_size = os.path.getsize(data_file)
        print(f"✓ Data file found: {data_file}")
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        return True
    else:
        print(f"✗ Data file not found: {data_file}")
        raise FileNotFoundError(f"Required data file not found: {data_file}")


check_data_availability = PythonOperator(
    task_id='check_data_availability',
    python_callable=check_data_file_exists,
    provide_context=True,
    dag=dag,
)


check_dvc_updates = BashOperator(
    task_id='check_dvc_updates',
    bash_command="""
    cd /opt/airflow && \
    if [ -f dvc.lock ]; then
        echo "DVC lock file found. Checking for updates..."
        dvc status || echo "DVC status check completed"
    else
        echo "No DVC lock file found"
    fi
    """,
    dag=dag,
)


prepare_data = BashOperator(
    task_id='prepare_data',
    bash_command="""
    cd /opt/airflow && \
    python src/prepare.py data/raw/dataset.csv data/prepared && \
    echo "Data preparation completed successfully"
    """,
    dag=dag,
)


train_model = BashOperator(
    task_id='train_model',
    bash_command="""
    cd /opt/airflow && \
    mkdir -p /tmp/mlflow_db && \
    export MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow_db/mlflow.db && \
    python src/train.py data/prepared data/models \
        --n-estimators 400 \
        --max-depth 12 \
        --min-samples-leaf 5 \
        --random-state 42 && \
    echo "Model training completed successfully"
    """,
    dag=dag,
)


def evaluate_model_performance(**context):
    """
    Evaluate model performance and decide whether to register it.
    Returns the task_id to execute next based on model quality.
    """
    metrics_path = '/opt/airflow/data/models/metrics.json'
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}")
        return 'notify_failure'
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"Model Metrics: {metrics}")
    
    r2_test = metrics.get('r2_test', 0)
    rmse_test = metrics.get('rmse_test', float('inf'))
    
    r2_threshold = 0.5
    rmse_threshold = 100
    
    print(f"R2 Score: {r2_test} (threshold: {r2_threshold})")
    print(f"RMSE: {rmse_test} (threshold: {rmse_threshold})")
    
    if r2_test >= r2_threshold and rmse_test <= rmse_threshold:
        print("✓ Model meets quality thresholds. Proceeding to registration.")
        return 'register_model'
    else:
        print("✗ Model does not meet quality thresholds. Skipping registration.")
        return 'notify_failure'


evaluate_and_branch = BranchPythonOperator(
    task_id='evaluate_and_branch',
    python_callable=evaluate_model_performance,
    provide_context=True,
    dag=dag,
)


register_model = BashOperator(
    task_id='register_model',
    bash_command="""
    cd /opt/airflow && \
    export MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow_db/mlflow.db && \
    python -c "
import mlflow
import os
import json

mlflow.set_tracking_uri('sqlite:////tmp/mlflow_db/mlflow.db')
mlflow.set_experiment('Bike_Sharing_RF_Experiment')

with open('data/models/metrics.json', 'r') as f:
    metrics = json.load(f)

runs = mlflow.search_runs(
    experiment_names=['Bike_Sharing_RF_Experiment'],
    order_by=['attribute.start_time DESC'],
    max_results=1
)

print(f'Found {len(runs)} run(s)')

if not runs.empty:
    run_id = runs.iloc[0]['run_id']
    print(f'Latest run_id: {run_id}')
    
    model_uri = f'runs:/{run_id}/random_forest_model'
    print(f'Model URI: {model_uri}')
    
    model_name = 'BikeSharing_RandomForest'
    
    try:
        model_version = mlflow.register_model(model_uri, model_name)
        print(f'✓ Model registered: {model_name} version {model_version.version}')
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )
        print(f'✓ Model transitioned to Staging stage')
        print(f'✓ Model {model_name} v{model_version.version} is now in Staging!')
        
    except Exception as e:
        print(f'Model registration failed: {e}')
        raise
else:
    print('No MLflow runs found')
    raise Exception('No runs available for model registration')
" && echo "✅ Model registered successfully in MLflow Model Registry"
    """,
    dag=dag,
)


notify_failure = EmptyOperator(
    task_id='notify_failure',
    dag=dag,
)


notify_success = EmptyOperator(
    task_id='notify_success',
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)


check_data_availability >> check_dvc_updates >> prepare_data >> train_model >> evaluate_and_branch

evaluate_and_branch >> register_model >> notify_success
evaluate_and_branch >> notify_failure >> notify_success
