from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import os
import pandas as pd


model=joblib.load('models/best_model.pkl')
app=FastAPI()

@app.get('/health')
def health_check():
    """check if the API is running"""

    return{
        'status': 'ok',
        'model': 'Customer-Churn-V1'
    }

class CustomerInput(BaseModel):
        tenure_months: int
        monthly_charges: float
        contract: str
        internet_service: str
        online_security: str
        tech_support: str
        payment_method: str
        senior_citizen: str
        partner: str
        dependents: str
        phone_service: str
        multiple_lines: str
        online_backup: str
        device_protection: str
        streaming_tv: str
        streaming_movies: str
        paperless_billing: str
        gender: str
        cltv: int 

class PredictionOutput(BaseModel):
       churn_probability: float
       churn_prediction: int
       risk_level: str


    
    
@app.post('/predict', response_model=PredictionOutput)


def predict(customer:CustomerInput):
      
    data=customer.model_dump()
    df=pd.DataFrame([data])

    from src.data import fix_dtypes,encode_features,scale_features
    from src.features import engineer_features


    df = df.rename(columns={
    'tenure_months': 'Tenure Months',
    'monthly_charges': 'Monthly Charges',
    'contract': 'Contract',
    'internet_service': 'Internet Service',
    'online_security': 'Online Security',
    'tech_support': 'Tech Support',
    'payment_method': 'Payment Method',
    'senior_citizen': 'Senior Citizen',
    'partner': 'Partner',
    'dependents': 'Dependents',
    'phone_service': 'Phone Service',
    'multiple_lines': 'Multiple Lines',
    'online_backup': 'Online Backup',
    'device_protection': 'Device Protection',
    'streaming_tv': 'Streaming TV',
    'streaming_movies': 'Streaming Movies',
    'paperless_billing': 'Paperless Billing',
    'gender': 'Gender',
    'cltv':'CLTV'})
    df['Churn Label']='No'
    df= engineer_features(df)
    df=fix_dtypes(df)
    df=encode_features(df)
    # Align columns with training data
    training_cols = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Tenure Months', 'Phone Service', 'Paperless Billing',
    'Monthly Charges', 'CLTV', 'charge_per_tenure',
    'risk_category', 'is_fully_loaded',
    'Multiple Lines_No phone service', 'Multiple Lines_Yes',
    'Internet Service_Fiber optic', 'Internet Service_No',
    'Online Security_No internet service', 'Online Security_Yes',
    'Online Backup_No internet service', 'Online Backup_Yes',
    'Device Protection_No internet service', 'Device Protection_Yes',
    'Tech Support_No internet service', 'Tech Support_Yes',
    'Streaming TV_No internet service', 'Streaming TV_Yes',
    'Streaming Movies_No internet service', 'Streaming Movies_Yes',
    'Contract_One year', 'Contract_Two year',
    'Payment Method_Credit card (automatic)',
    'Payment Method_Electronic check', 'Payment Method_Mailed check']

# Add missing columns with 0
    for col in training_cols:
      if col not in df.columns:
         df[col] = 0
    df = df[training_cols]
    df=scale_features(df)
    
    proba=model.predict_proba(df)[0][1]
    prediction=int(proba>=0.5)

    if proba >= 0.7:
        risk = 'High Risk'
    elif proba >= 0.4:
            risk = 'Medium Risk'
    else:
            risk = 'Low Risk'

    return PredictionOutput(churn_probability=round(float(proba),4),churn_prediction=prediction, risk_level=risk)

    