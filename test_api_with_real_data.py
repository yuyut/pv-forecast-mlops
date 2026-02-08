import requests
import pandas as pd
import json
from pathlib import Path

# Load test data
test_parquet = Path("data/test.parquet")
if not test_parquet.exists():
    raise FileNotFoundError(f"Test data not found at {test_parquet}")

# Read parquet file
df_test = pd.read_parquet(test_parquet)

print(f"Loaded test data: {df_test.shape}")
print(f"Columns: {df_test.columns.tolist()}\n")

# Sample 2-3 random rows
sample_df = df_test.sample(n=min(3, len(df_test)), random_state=42)

# Keep only the minimal required columns for API to process
# The API will do the feature engineering
required_cols = ['timestamp', 'latitude', 'longitude', 'dc_capacity', 'ac_capacity', 
                 'temperature_2m', 'relative_humidity_2m', 'shortwave_radiation',
                 'direct_radiation', 'diffuse_radiation', 'direct_normal_irradiance',
                 'wind_speed_10m', 'wind_direction_10m', 'cloud_cover', 'precipitation',
                 'is_day', 'sunshine_duration', 'tilt', 'azimuth']

# Get only columns that exist
input_cols = [col for col in required_cols if col in sample_df.columns]
input_data = sample_df[input_cols].fillna(0)

# Convert timestamps to strings for JSON serialization
if 'timestamp' in input_data.columns:
    input_data['timestamp'] = input_data['timestamp'].astype(str)

# Convert to list of dicts for API
mock_rows = input_data.to_dict('records')

# Create request payload
payload = {"rows": mock_rows}

print("Sample data (first row):")
print(json.dumps(mock_rows[0], indent=2))
print(f"\nSending {len(mock_rows)} rows to API...\n")

# Test health
try:
    response = requests.get("http://127.0.0.1:8000/health")
    print(f"✅ Health check: {response.json()}")
except Exception as e:
    print(f"❌ Health check failed: {e}")
    exit(1)

# Test prediction
try:
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Predictions received:")
        for i, pred in enumerate(result.get('predictions', [])):
            print(f"   Row {i+1}: {pred:.4f} kW")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Prediction failed: {e}")
