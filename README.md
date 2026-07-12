# Customer Churn Prediction — MLOps Platform

Predicts telecom customer churn using machine learning, 
served via a REST API with a Streamlit dashboard.

## Status
🔄 In Progress — Models Trained and Tuned

## Results
| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.8091 | 0.6084 | 0.8516 |
| Random Forest | 0.7999 | 0.5913 | 0.8449 |
| XGBoost (baseline) | 0.7949 | 0.5912 | 0.8458 |
| XGBoost (tuned) ⭐ | 0.8055 | 0.6017 | 0.8522 |

## Project Structure
customer-churn-mlops/
├── data/                  # Dataset and EDA output plots
├── notebooks/             # Exploratory analysis
│   └── 01_EDA.ipynb
├── src/                   # Production Python modules
│   ├── data.py            # Data loading and preprocessing
│   ├── features.py        # Feature engineering
│   ├── split.py           # Train/test split and cross validation
│   ├── train.py           # Model training and evaluation
│   └── tune.py            # Hyperparameter tuning with Optuna
├── models/                # Saved model files (gitignored)
├── requirements.txt
└── README.md

## Dataset
IBM Telco Customer Churn — Extended Version
7,043 customers · 33 features
Source: https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

## Key EDA Findings
- 26.5% churn rate — moderate class imbalance
- Contract type is the strongest predictor — month-to-month customers 
  churn at 42% vs 3% for two-year contracts
- Customers without Tech Support and Online Security churn significantly more
- Short tenure + high monthly charges = highest risk profile

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · Optuna · 
Weights & Biases · FastAPI (coming) · Docker (coming) · Streamlit (coming)

## Experiment Tracking
All experiments tracked on Weights & Biases:
https://wandb.ai/alijankhan407-air-university/customer-churn

## Author
Mahmood Ali Khan — M.Sc. Artificial Intelligence, Air University Islamabad 
