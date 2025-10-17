# src/model_training.py
import yaml
import pandas as pd
import pickle
import json
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(params_path):
    logging.info("Starting model training...")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    train_params = params['model_training']
    target_col = params['data_processing']['target_column']
    
    processed_data_path = Path("data/processed/train.csv")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    logging.info(f"Loading training data from {processed_data_path}")
    train_df = pd.read_csv(processed_data_path)

    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]

    model = LogisticRegression(
        random_state=train_params['random_state'],
        max_iter=train_params['max_iter']
    )
    
    logging.info("Fitting the model...")
    model.fit(X_train, y_train)

    # Save model and columns
    with open(models_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(models_dir / "model_columns.json", "w") as f:
        json.dump({"columns": X_train.columns.tolist()}, f)

    logging.info(f"Model and columns saved to {models_dir}")

if __name__ == "__main__":
    train_model("params.yaml")