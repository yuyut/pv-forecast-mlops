from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="pv_train_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # trigger manually
    catchup=False,
) as dag:

    train = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow && python -m src.train",
        env={
            # point training to MLflow container
            "MLFLOW_TRACKING_URI": "http://mlflow:5000"
        }
    )

    train
