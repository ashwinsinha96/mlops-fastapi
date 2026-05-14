from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI()

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Home route
@app.get("/")
def home():
    return {"message": "FastAPI ML Model Deployment Running"}

# Prediction route
@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}