"""Real-time prediction service"""
from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

class DemandPredictor:
    def __init__(self, model_uri):
        self.model = mlflow.pyfunc.load_model(model_uri)
    
    def predict(self, features: dict) -> float:
        """Generate prediction"""
        df = pd.DataFrame([features])
        prediction = self.model.predict(df)
        return float(prediction[0])

predictor = DemandPredictor("models:/demand-forecast/production")

@app.post("/predict")
async def predict(features: dict):
    """Prediction endpoint"""
    prediction = predictor.predict(features)
    return {"forecast": prediction}
