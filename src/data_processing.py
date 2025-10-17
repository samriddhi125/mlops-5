# src/data_processing.py
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(config_path):
    logging.info("Starting data processing...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    params = config['data_processing']
    raw_data_path = Path("data/raw/fatalities_isr_pse_conflict_2000_to_2023.csv")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    # Basic Cleaning
    df = df.drop(columns=params['features_to_drop'])
    df = df.dropna(subset=[params['target_column'], 'age']) # Drop rows where target or age is null
    df = df.fillna('Unknown') # Fill remaining NaNs

    # Feature Engineering (One-Hot Encode categorical features)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.drop(params['target_column'])
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    logging.info("Splitting data into training and testing sets.")
    train_df, test_df = train_test_split(
        df_encoded,
        test_size=params['test_size'],
        random_state=params['random_state']
    )

    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    logging.info(f"Processed data saved to {processed_dir}")

if __name__ == "__main__":
    process_data("params.yaml")