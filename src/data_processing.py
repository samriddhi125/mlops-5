# src/data_processing.py
import pandas as pd
from pathlib import Path

def process_data_for_forecasting():
    raw_data_path = Path("data/raw/fatalities_isr_pse_conflict_2000_to_2023.csv")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_data_path)
    df['date_of_event'] = pd.to_datetime(df['date_of_event'])

    df_palestinian = df[df['citizenship'] == 'Palestinian'].copy()

    # Aggregate by month to create the time series
    time_series_df = df_palestinian.set_index('date_of_event').resample('M').size().reset_index(name='fatalities')
    time_series_df.rename(columns={'date_of_event': 'date'}, inplace=True)

    # Save the processed time series data
    output_path = processed_dir / "monthly_fatalities_timeseries.csv"
    time_series_df.to_csv(output_path, index=False)
    print(f"Time series data saved to {output_path}")

if __name__ == "__main__":
    process_data_for_forecasting()