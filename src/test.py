# src/test.py
import os
from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ARTIFACT_DIR = Path("artifacts")

@dataclass
class TestConfig:
    test_data_path: str = "data/test.parquet"
    model_path: str = "artifacts/pv_pipeline.joblib"
    target_col: str = "y"
    time_col: str = "timestamp"

def load_test_data(cfg: TestConfig) -> pd.DataFrame:
    """Load test data from parquet file."""
    df = pd.read_parquet(cfg.test_data_path)
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    df = df.sort_values(cfg.time_col).reset_index(drop=True)
    return df

def load_model(cfg: TestConfig):
    """Load trained model pipeline."""
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"Model not found at {cfg.model_path}. Train the model first using train.py")
    return joblib.load(cfg.model_path)

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

def main():
    cfg = TestConfig()

    print("=" * 60)
    print("PV Forecast Model Testing")
    print("=" * 60)

    # Load model
    print(f"\n1. Loading model from {cfg.model_path}...")
    pipeline = load_model(cfg)
    print("   ✓ Model loaded successfully")

    # Load test data
    print(f"\n2. Loading test data from {cfg.test_data_path}...")
    df = load_test_data(cfg)
    print(f"   ✓ Test data loaded: {len(df)} samples")

    # Prepare features and target
    print("\n3. Preparing features...")
    cols_to_drop = [cfg.target_col, cfg.time_col, 'unique_id', 'dataset']

    y_test = df[cfg.target_col].values
    X_test = df.drop(columns=cols_to_drop)

    print(f"   ✓ Features prepared: {X_test.shape[1]} features")
    print(f"   Feature columns: {', '.join(X_test.columns[:5].tolist())}...")

    # Make predictions
    print("\n4. Making predictions...")
    y_pred = pipeline.predict(X_test)
    print("   ✓ Predictions completed")

    # Evaluate
    print("\n5. Evaluating model performance...")
    metrics = evaluate_model(y_test, y_pred)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"MAE (Mean Absolute Error):     {metrics['MAE']:.6f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:.6f}")
    print(f"R² Score:                      {metrics['R2']:.6f}")
    print(f"MAPE (Mean Absolute % Error):  {metrics['MAPE']:.2f}%")
    print("=" * 60)

    # Sample predictions
    print("\n6. Sample predictions (first 10):")
    print("-" * 60)
    print(f"{'Index':<8} {'Actual':<15} {'Predicted':<15} {'Error':<15}")
    print("-" * 60)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_pred[i])
        print(f"{i:<8} {y_test[i]:<15.6f} {y_pred[i]:<15.6f} {error:<15.6f}")
    print("-" * 60)

    # Save predictions
    results_df = df[[cfg.time_col, 'unique_id', cfg.target_col]].copy()
    results_df['y_pred'] = y_pred
    results_df['error'] = np.abs(y_test - y_pred)
    results_df['error_pct'] = (results_df['error'] / (results_df[cfg.target_col] + 1e-8)) * 100

    output_path = Path("artifacts") / "test_predictions.parquet"
    results_df.to_parquet(output_path, index=False)
    print(f"\n7. Predictions saved to {output_path}")

    print("\n✓ Testing completed successfully!\n")

if __name__ == "__main__":
    main()
