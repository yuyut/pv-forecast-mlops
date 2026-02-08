# src/train.py
import os
from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

@dataclass
class TrainConfig:
    data_path: str = "data/train.parquet"
    val_data_path: str = "data/val.parquet"
    target_col: str = "y"
    time_col: str = "timestamp"
    experiment_name: str = "pv_forecasting"
    model_name: str = "xgb_regressor"
    n_splits: int = 3
    sample_frac: float = 1.0  # Use 1.0 for full data, 0.1 for 10% (faster training)

def load_data(cfg: TrainConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.data_path)
    # ensure datetime (already datetime64 in parquet)
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    df = df.sort_values(cfg.time_col).reset_index(drop=True)
    return df

def split_train_val_time(df: pd.DataFrame, cfg: TrainConfig):
    # Simple final-holdout: last 20% as validation
    cut = int(len(df) * 0.8)
    train_df = df.iloc[:cut]
    val_df = df.iloc[cut:]
    return train_df, val_df

def build_pipeline(num_cols):
    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )

    model = XGBRegressor(
        n_estimators=300,  # Reduced from 800
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
        verbosity=1  # Show progress
    )

    pipeline = Pipeline([
        ("preproc", preproc),
        ("model", model)
    ])
    return pipeline

def main():
    cfg = TrainConfig()

    mlflow.set_experiment(cfg.experiment_name)

    # Load training data
    train_df = load_data(cfg)

    # Optional: Sample data for faster training during development
    if cfg.sample_frac < 1.0:
        print(f"\n Using {cfg.sample_frac*100:.0f}% of data for faster training...")
        train_df = train_df.sample(frac=cfg.sample_frac, random_state=42).reset_index(drop=True)
        print(f"   Sampled {len(train_df):,} rows")

    # Load validation data from separate file
    print(f"Loading validation data from {cfg.val_data_path}...")
    val_df = pd.read_parquet(cfg.val_data_path)
    val_df[cfg.time_col] = pd.to_datetime(val_df[cfg.time_col])
    val_df = val_df.sort_values(cfg.time_col).reset_index(drop=True)

    # Define columns to drop (non-feature columns)
    cols_to_drop = [cfg.target_col, cfg.time_col, 'unique_id', 'dataset']

    y_train = train_df[cfg.target_col].values
    X_train = train_df.drop(columns=cols_to_drop)

    y_val = val_df[cfg.target_col].values
    X_val = val_df.drop(columns=cols_to_drop)

    pipeline = build_pipeline(num_cols=X_train.columns.tolist())

    with mlflow.start_run(run_name=cfg.model_name):
        # Log config + feature list
        mlflow.log_param("model", "XGBRegressor")
        mlflow.log_param("n_estimators", pipeline.named_steps["model"].n_estimators)
        mlflow.log_param("max_depth", pipeline.named_steps["model"].max_depth)
        mlflow.log_param("learning_rate", pipeline.named_steps["model"].learning_rate)
        mlflow.log_param("subsample", pipeline.named_steps["model"].subsample)
        mlflow.log_param("colsample_bytree", pipeline.named_steps["model"].colsample_bytree)
        mlflow.log_param("features", ",".join(X_train.columns))

        print(f"\nTraining on {len(X_train):,} samples with {X_train.shape[1]} features...")
        print(f"Validation set: {len(X_val):,} samples")

        # Fit preprocessing and transform validation data for early stopping
        X_train_transformed = pipeline.named_steps["preproc"].fit_transform(X_train)
        X_val_transformed = pipeline.named_steps["preproc"].transform(X_val)

        # Train model with early stopping on transformed data
        pipeline.named_steps["model"].fit(
            X_train_transformed, y_train,
            eval_set=[(X_val_transformed, y_val)],
            verbose=50  # Print every 50 rounds
        )

        pred = pipeline.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        rmse = mean_squared_error(y_val, pred, squared=False)

        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmse", rmse)

        # Save & log artifact
        model_path = ARTIFACT_DIR / "pv_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))

        # Optional: log model in MLflow format too
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"Saved model to {model_path}")
        print(f"VAL MAE={mae:.4f}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()
