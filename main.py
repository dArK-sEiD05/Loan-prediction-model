
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Optional
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

app = FastAPI(title="Loan Eligibility & Risk Scoring API")

# Pydantic model for input validation
class BorrowerFeatures(BaseModel):
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: int
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    HasMortgage: str
    HasCoSigner: str
    HasDependents: str
    Age: float
    EmploymentType: str
    LoanPurpose: str

# Load artifacts
cat_imputer = joblib.load('artifacts/cat_imputer.joblib')
scaler = joblib.load('artifacts/scaler.joblib')
xgb_model = joblib.load('artifacts/xgb_model.joblib')
model_metrics = joblib.load('artifacts/model_metrics.joblib')
features = joblib.load('artifacts/features.joblib')

@app.post("/predict")
async def predict(input_data: BorrowerFeatures):
    """Predict loan default risk score for a borrower."""
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([input_data.dict()])
        cat_cols = ['LoanPurpose', 'HasDependents', 'HasMortgage', 'HasCoSigner']
        
        # Impute categorical columns
        data[cat_cols] = cat_imputer.transform(data[cat_cols])
        
        # Map categorical columns
        categorical_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
        for col in categorical_cols:
            data[col] = data[col].map({'Yes': 1, 'No': 0})
        
        # One-hot encode EmploymentType and ensure all categories
        emp_categories = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
        data = pd.get_dummies(data, columns=['EmploymentType'], prefix='Emp', dtype=int)
        for cat in emp_categories:
            col_name = f'Emp_{cat}'
            if col_name not in data.columns:
                data[col_name] = 0
        
        # Feature engineering
        data['AgeRisk'] = np.where((data['Age'] < 25) | (data['Age'] > 60), 1, 0)
        data['IncomeRisk'] = data['Income'].max() - data['Income']
        data['CreditScoreRisk'] = data['CreditScore'].max() - data['CreditScore']
        data['MonthsEmployedRisk'] = data['MonthsEmployed'].max() - data['MonthsEmployed']
        r = data['InterestRate'] / 100.0 / 12.0
        n = data['LoanTerm'].replace(0, 1)
        P = data['LoanAmount']
        data['MonthlyPayment'] = np.where(
            np.isclose(r, 0),
            P / n,
            P * r / (1 - (1 + r) ** (-n))
        )
        data['PaymentToIncome'] = data['MonthlyPayment'] / (data['Income'] / 12 + 1e-9)
        bins = [0, 580, 670, 740, 800, 1000]
        labels = ['poor', 'fair', 'avg', 'good', 'excellent']
        data['CreditScore_bin'] = pd.cut(data['CreditScore'], bins=bins, labels=labels, include_lowest=True)
        ord_map = {'poor': 0, 'fair': 1, 'avg': 2, 'good': 3, 'excellent': 4}
        data['CreditScore_num'] = data['CreditScore_bin'].map(ord_map).astype(float).fillna(data['CreditScore'].median())
        data['EmploymentStabilityFlag'] = (data['MonthsEmployed'] < 12).astype(int)
        
        # Select and align features
        data = data.reindex(columns=features, fill_value=0)
        
        # Debug: Verify feature count
        if data.shape[1] != len(features):
            raise ValueError(f"Expected {len(features)} features, got {data.shape[1]}: {data.columns.tolist()}")
        
        # Scale
        data_scaled = scaler.transform(data)
        
        pred=xgb_model.predict(data_scaled)
        # Predict
        risk_prob = xgb_model.predict_proba(data_scaled)[0][1]
        return {"risk_score": float(risk_prob),
                "output":int(pred[0]),
                "label": "Default" if int(pred[0]) == 1 else "Non-Default"
}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model_info")
async def model_info():
    """Return model performance metrics."""
    try:
        return {
            "metrics": model_metrics,
            "features": features,
            "features_importance":xgb_model.feature_importances_.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
