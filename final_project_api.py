# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("final_project_api")

# Create input/output pydantic models
input_model = create_model("final_project_api_input", **{'Age': 25.0, 'Cholesterol': 163.0, 'Heart Rate': 49.0, 'Diabetes': 1.0, 'Family History': 1.0, 'Smoking': 1.0, 'Obesity': 0.0, 'Alcohol Consumption': 0.0, 'Exercise Hours Per Week': 16.507381439208984, 'Previous Heart Problems': 1.0, 'Medication Use': 1.0, 'Stress Level': 1.0, 'Sedentary Hours Per Day': 8.181629180908203, 'Income': 42197.0, 'BMI': 21.471824645996094, 'Triglycerides': 290.0, 'Physical Activity Days Per Week': 5.0, 'Sleep Hours Per Day': 10.0, 'Sistolic': 141.0, 'Distolic': 70.0, 'Sex_Female': 0.0, 'Sex_Male': 1.0, 'Diet_Average': 0.0, 'Diet_Healthy': 1.0, 'Diet_Unhealthy': 0.0})
output_model = create_model("final_project_api_output", prediction=0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
