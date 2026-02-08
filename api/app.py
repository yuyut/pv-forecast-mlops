from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import os

# Get the parent directory of this script
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "pv_pipeline.joblib"

app = FastAPI(title="PV Day-Ahead Forecast API")

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]

    model_config = {
        "json_schema_extra": {
            "example": {
                "rows": [
                    {
                        "timestamp": "2024-02-08T12:00:00",
                        "latitude": 52.1,
                        "longitude": 5.1,
                        "dc_capacity": 5.0,
                        "ac_capacity": 4.0,
                        "temperature_2m": 15.0,
                        "relative_humidity_2m": 80.0,
                        "shortwave_radiation": 500.0,
                        "wind_speed_10m": 5.0,
                        "cloud_cover": 50.0,
                        "precipitation": 0.0
                    }
                ]
            }
        }
    }

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Train first.")
    model = joblib.load(MODEL_PATH)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as in training.
    Converts raw input to model-ready features.
    """
    df = df.copy()
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour.astype('int8')
        df['day_of_week'] = df['timestamp'].dt.dayofweek.astype('int8')
        df['day_of_year'] = df['timestamp'].dt.dayofyear.astype('int16')
        df['month'] = df['timestamp'].dt.month.astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype('float32')
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype('float32')
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25).astype('float32')
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25).astype('float32')
        
        # Solar position proxy
        df['hours_from_noon'] = np.abs(df['hour'] - 12).astype('int8')
        
        # Assume is_day based on hour if not provided
        if 'is_day' not in df.columns:
            df['is_day'] = ((df['hour'] >= 6) & (df['hour'] < 18)).astype('int8')
    
    # Engineered interactions
    if 'shortwave_radiation' in df.columns and 'is_day' in df.columns:
        df['radiation_day'] = (df['shortwave_radiation'] * df['is_day']).astype('float32')
    
    # Capacity factor (will be 0 for prediction since we don't have y yet)
    if 'dc_capacity' in df.columns:
        df['capacity_factor'] = 0.0
    
    # Lag features (set to 0 for first prediction - user should provide if they have history)
    df['y_lag_1'] = df.get('y_lag_1', 0.0)
    df['y_lag_2'] = df.get('y_lag_2', 0.0)
    df['y_lag_6'] = df.get('y_lag_6', 0.0)
    df['y_lag_48'] = df.get('y_lag_48', 0.0)
    df['y_roll_mean_12'] = df.get('y_roll_mean_12', 0.0)
    df['y_roll_mean_48'] = df.get('y_roll_mean_48', 0.0)
    
    # Fill missing values
    df = df.fillna(0)
    
    return df

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.rows)
    
    # Apply feature engineering
    df = engineer_features(df)
    
    # Predict
    preds = model.predict(df).tolist()
    return {"predictions": preds}
