AI-Powered Loan Eligibility & Risk Scoring System

This project implements a machine learning system to predict loan default risk using an XGBoost model, served via a FastAPI backend. It includes data preprocessing, feature engineering, model training, evaluation, and visualizations to provide insights into loan risk factors. The system is designed to assist in loan eligibility decisions by generating a risk score for borrowers.

Project Overview





Objective: Predict the probability of loan default based on borrower features.



Model: XGBoost classifier with SMOTE and RandomUnderSampler to handle class imbalance.



Features: 19 features, including engineered features like IncomeRisk, CreditScoreRisk, and PaymentToIncome.



API: FastAPI endpoints for prediction (/predict) and model information (/model_info).



Visualizations: Correlation heatmap, feature distributions, class imbalance, risk segmentation, ROC curve, precision-recall curve, and feature importance plot.

Setup

Prerequisites





Python 3.8+



Git



Jupyter Notebook (optional, for EDA)

Installation





Clone the Repository:

git clone <your-repo-url>
cd loan-risk-scoring



Install Dependencies:

pip install -r requirements.txt



Set Up Environment Variables: Create a .env file in the root directory:

API_KEY=your_secure_api_key
PORT=8000

Directory Structure

loan-risk-scoring/
├── artifacts/
│   ├── cat_imputer.joblib
│   ├── scaler.joblib
│   ├── xgb_model.joblib
│   ├── model_metrics.joblib
│   ├── y_test.joblib
│   ├── y_prob.joblib
│   ├── features.joblib
│   ├── feature_importance.joblib
├── plots/
│   ├── correlation_heatmap.png
│   ├── class_imbalance.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   └── *_distribution.png
│   └── *_segmentation.png
├── notebooks/
│   └── eda_and_training.ipynb
├── app.py
├── train_and_export.py
├── visualizations.py
├── requirements.txt
├── .env
└── README.md

Running the API





Start the FastAPI Server:

python app.py

Or, for production:

gunicorn -k uvicorn.workers.UvicornWorker app:app -w 4 --bind 0.0.0.0:8000



API Endpoints:





POST /predict: Predict loan default risk score.





Input (JSON):

{
  "Income": 55000,
  "LoanAmount": 12000,
  "CreditScore": 680,
  "MonthsEmployed": 24,
  "NumCreditLines": 2,
  "InterestRate": 7.0,
  "LoanTerm": 48,
  "DTIRatio": 0.4,
  "HasMortgage": "Yes",
  "HasCoSigner": "No",
  "HasDependents": "No",
  "Age": 28,
  "EmploymentType": "Full-time",
  "LoanPurpose": "Auto"
}



Output (example):

{"risk_score": 0.27}



GET /model_info: Retrieve model performance metrics, features, and feature importance.





Output (example):

{
  "metrics": {
    "accuracy": 0.88,
    "classification_report": {...},
    "roc_auc": 0.92,
    "confusion_matrix": [[60075, 7606], [5738, 3155]]
  },
  "features": [
    "IncomeRisk", "LoanAmount", "CreditScoreRisk", "MonthsEmployedRisk",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "HasMortgage", "HasCoSigner", "AgeRisk", "HasDependents",
    "Emp_Full-time", "Emp_Self-employed", "Emp_Unemployed", "Emp_Part-time",
    "PaymentToIncome", "CreditScore_num", "EmploymentStabilityFlag"
  ],
  "feature_importance": {
    "IncomeRisk": 0.12,
    "LoanAmount": 0.08,
    "CreditScoreRisk": 0.15,
    "MonthsEmployedRisk": 0.09,
    "NumCreditLines": 0.06,
    "InterestRate": 0.07,
    "LoanTerm": 0.05,
    "DTIRatio": 0.10,
    "HasMortgage": 0.03,
    "HasCoSigner": 0.04,
    "AgeRisk": 0.02,
    "HasDependents": 0.03,
    "Emp_Full-time": 0.04,
    "Emp_Self-employed": 0.03,
    "Emp_Unemployed": 0.03,
    "Emp_Part-time": 0.02,
    "PaymentToIncome": 0.11,
    "CreditScore_num": 0.13,
    "EmploymentStabilityFlag": 0.05
  }
}

Retraining the Model





Prepare Dataset:





Place your dataset (loan_dataset.csv) in the root directory or update the path in train_and_export.py.



Run Training Script:

python train_and_export.py

This generates artifacts in the artifacts/ directory:





cat_imputer.joblib: Imputer for categorical columns.



scaler.joblib: StandardScaler for numerical features.



xgb_model.joblib: Trained XGBoost model.



model_metrics.joblib: Model performance metrics.



y_test.joblib: Test set labels.



y_prob.joblib: Test set probability predictions.



features.joblib: List of 19 features.



feature_importance.joblib: Feature importance scores.



Run Jupyter Notebook (Optional):





Open notebooks/eda_and_training.ipynb for exploratory data analysis and training.



Update the dataset path and run all cells.

Generating Visualizations

Run the visualization script to generate plots:

python visualizations.py

Visualization Outputs





Correlation Heatmap (plots/heatmap.png): Shows correlations between numerical features and the target (Default).






Class Imbalance (plots/classimbalance.png): Bar plot showing the distribution of Default (0 = No, 1 = Yes).



Risk Segmentation (plots/risk_segment.png): Box plots for CreditScoreRisk, PaymentToIncome, and IncomeRisk by default status.




Precision-Recall Curve (plots/precision_recall_curve.png): Precision vs. recall for model performance.



Feature Importance (plots/feature_importance.png): Bar plot of feature importance scores from the XGBoost model.