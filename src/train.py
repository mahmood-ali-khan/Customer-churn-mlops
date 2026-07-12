import wandb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from src.data import load_data, fix_dtypes, encode_features, scale_features
from src.features import engineer_features
from src.split import split_data

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: trained sklearn model
        X_test (pd.DataFrame): test features
        y_test (pd.Series): test target
    
    Returns:
        dict: accuracy, f1, auc_roc scores
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }

def train_all():
    """Load data, train 3 models, return trained models and test data."""
    
    # Load and preprocess
    df = load_data('data/churn.xlsx')
    df = engineer_features(df)
    df = fix_dtypes(df)
    df = encode_features(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Model 1 — Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    print("Logistic Regression trained")
    lr_metrics = evaluate_model(lr, X_test, y_test)
    print(f"LR     — Accuracy: {lr_metrics['accuracy']:.4f} | F1: {lr_metrics['f1']:.4f} | AUC-ROC: {lr_metrics['auc_roc']:.4f}")
    wandb.init(project='customer-churn', name='logistic_regression')
    wandb.log(lr_metrics)
    wandb.finish()
    # Model 2 — Random Forest
    rf = RandomForestClassifier(n_estimators=45, random_state=42,max_depth=15,)
    rf.fit(X_train, y_train)
    print("Random Forest trained")
    rf_metrics = evaluate_model(rf, X_test, y_test)
    print(f"RF     — Accuracy: {rf_metrics['accuracy']:.4f} | F1: {rf_metrics['f1']:.4f} | AUC-ROC: {rf_metrics['auc_roc']:.4f}")
    wandb.init(project='customer-churn', name='random_forest')
    wandb.log(rf_metrics)
    wandb.finish()
    # Model 3 — XGBoost
    xgb = XGBClassifier(n_estimators=45, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    print("XGBoost trained")
    xgb_metrics = evaluate_model(xgb, X_test, y_test)
    print(f"XGBoost — Accuracy: {xgb_metrics['accuracy']:.4f} | F1: {xgb_metrics['f1']:.4f} | AUC-ROC: {xgb_metrics['auc_roc']:.4f}")
    wandb.init(project='customer-churn', name='xgboost')
    wandb.log(xgb_metrics)
    wandb.finish()
    return lr, rf, xgb, X_test, y_test


if __name__ == '__main__':
    lr, rf, xgb, X_test, y_test = train_all()



# After Logistic Regression

# After Random Forest


# After XGBoost
