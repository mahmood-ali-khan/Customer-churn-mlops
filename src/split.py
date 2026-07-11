import pandas as pd
from  sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

def split_data (df, target='churn', test_size=0.2, random_state=42):
    """
    Split DataFrame into train and test sets.
    
    Args:
        df (pd.DataFrame): fully preprocessed DataFrame
        target (str): target column name
        test_size (float): proportion for test set
        random_state (int): reproducibility seed
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X=df.drop(target, axis=1)
    y=df[target]
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=test_size, random_state=random_state, stratify=y)

    return  X_train, X_test, y_train, y_test

def validate_split (X_train, X_test, y_train, y_test):
    """
    Validate train/test split by printing key statistics.
    
    Args:
        X_train, X_test: feature splits
        y_train, y_test: target splits
    
    Returns:
        None
    """
    print("\n=== Split Validation ===")
    print(f"Training set:   {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test set:       {X_test.shape[0]} rows, {X_test.shape[1]} features")
    print(f"Train churn rate: {y_train.mean() * 100:.2f}%")
    print(f"Test churn rate:  {y_test.mean() * 100:.2f}%")
    print(f"Features: {X_train.shape[1]}")


def cross_validate(X_train, y_train, n_splits=5):
    """
    Run stratified k-fold cross validation.
    
    Args:
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target
        n_splits (int): number of folds
    
    Returns:
        None
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        LogisticRegression(max_iter=1000),
        X_train, y_train,
        cv=skf,
        scoring='f1'
    )
    
    print("\n=== Stratified K-Fold Cross Validation ===")
    print(f"Folds: {n_splits}")
    print(f"F1 scores: {scores.round(4)}")
    print(f"Mean F1:   {scores.mean():.4f}")
    print(f"Std:       {scores.std():.4f}")