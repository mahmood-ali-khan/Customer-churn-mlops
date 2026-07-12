from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_random_forest(X_train, y_train, n_trials=50):
    """
    Tune Random Forest hyperparameters using Optuna.
    
    Args:
        X_train: training features
        y_train: training target
        n_trials (int): number of Optuna trials
    
    Returns:
        dict: best hyperparameters
    """
    def objective(trial):
        n_estimators=trial.suggest_int('n_estimators',30,200)
        max_depth=trial.suggest_int('max_depth',5,30) 
        min_samples_split=trial.suggest_int('min_samples_split',2,10)
        min_samples_leaf=trial.suggest_int('min_samples_leaf',1,5)
    
        model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, random_state=42)
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
        return scores.mean()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n=== Random Forest Tuning Complete ===")
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params    



def tune_xgboost(X_train, y_train, n_trials=50):
    """
    Tune xgboost hyperparameters using Optuna.
    
    Args:
        X_train: training features
        y_train: training target
        n_trials (int): number of Optuna trials
    
    Returns:
        dict: best hyperparameters
    """
    def objective(trial):
        n_estimators=trial.suggest_int('n_estimators',30,200)
        max_depth=trial.suggest_int('max_depth',5,30) 
        learning_rate=trial.suggest_float('learning_rate',0.01,0.3)
        subsample=trial.suggest_float('subsample',0.6,1.0)
    
        model=XGBClassifier(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample, random_state=42,eval_metric='logloss')
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
        return scores.mean()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n=== xgboost Tuning Complete ===")
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params    