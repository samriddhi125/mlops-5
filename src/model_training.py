# src/model_training.py
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from pathlib import Path

def train_forecasting_model():
    data_path = Path("data/processed/monthly_fatalities_timeseries.csv")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path, index_col='date', parse_dates=True)
    
    # A simple SARIMA model configuration (p,d,q)(P,D,Q,s)
    # These parameters can be tuned for better performance
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12) # 12 for monthly seasonality

    print("Training SARIMA model...")
    model = SARIMAX(df['fatalities'], order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # Save the trained model
    with open(model_dir / "forecasting_model.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"Forecasting model trained and saved to {model_dir}")

if __name__ == "__main__":
    train_forecasting_model()