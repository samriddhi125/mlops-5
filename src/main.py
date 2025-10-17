# src/main.py
import pickle
import json
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

app = FastAPI(title="Israel-Palestine Conflict Fatality Predictor")

# Load model and columns
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/model_columns.json", "r") as f:
    model_columns = json.load()['columns']

@app.get("/", response_class=HTMLResponse)
def read_root():
    # Simple form for user input
    return """
    <html>
        <head><title>Fatality Predictor</title></head>
        <body>
            <h1>Predict Citizenship</h1>
            <form action="/predict/" method="post">
                Age: <input type="number" name="age" value="25"><br>
                Gender: <select name="gender"><option value="M">Male</option><option value="F">Female</option></select><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict/")
def predict(age: int = Form(...), gender: str = Form(...)):
    # Create a DataFrame for a single prediction
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0 # Initialize with zeros
    
    # Set the known values
    input_data.at[0, 'age'] = age
    if gender.upper() == 'M':
        if 'gender_M' in input_data.columns:
            input_data.at[0, 'gender_M'] = 1
    
    # Ensure all columns match the training data, filling missing ones with 0
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)
    confidence = probabilities.max()

    return {
        "predicted_citizenship": prediction,
        "confidence_score": float(confidence)
    }