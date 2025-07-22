import joblib
import pandas as pd
import numpy as np
import os

class ChurnModel:
    def __init__(self, model_path="models/churn_model.pkl"):
        # Load the model (which includes the preprocessor in the pipeline)
        self.model = joblib.load(model_path)
        
    def predict(self, input_data):
        """Make prediction with the model"""
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        churn_prob = self.model.predict_proba(input_df)[0, 1]
        churn_pred = churn_prob >= 0.5
        
        return {
            "churn_probability": float(churn_prob),
            "churn_prediction": bool(churn_pred)
        } 
