import pandas as pd 


def add_charge_per_tenure(df):
    """
    Adding Charges per tenure month feature.
    
    Args: 
     df(pd.Dataframe):Dataframe after preprocessing 

    Returns:
     DataFrame with Charge per tenure column added.

    """

    df['charge_per_tenure']=df['Monthly Charges']/(df['Tenure Months']+1)

    return df

def add_risk_category(df):
    """
    Adding Risk category feature
    Args:
     df(pd.Dataframe):Dataframe after adding charge per tenure feature

    Returns:
      DataFrame with risk category column added.
    """
    def categorise(row):
        if row['Tenure Months']<12 and row['Monthly Charges']>70:
            return 'High Risk'
        elif row['Tenure Months'] < 24:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    df['risk_category']=df.apply(categorise, axis=1)

    return df

def add_is_fully_loaded(df):
    """
    Add binary flag for customers with multiple services.
    
    Args:
        df (pd.DataFrame): DataFrame after add_risk_category
    
    Returns:
        pd.DataFrame: DataFrame with is_fully_loaded column added
    """
    def check_services(row):
        if (row['Phone Service'] == 'Yes' and row['Internet Service'] != 'No' and (row['Tech Support'] == 'Yes' or row['Online Security'] == 'Yes')): 
            return 1
        else:
            return 0
    df['is_fully_loaded']=df.apply(check_services, axis=1)
    return df

def print_pivot_summary(df):
    """
    Print pivot table of average Monthly Charges 
    by Contract type and Churn status.
    
    Args:
        df (pd.DataFrame): DataFrame before encoding
    
    Returns:
        None
    """
    pivot=pd.pivot_table(df, values='Monthly Charges', index='Contract',
    columns='Churn Label',aggfunc='mean')
    print("\n=== Average Monthly Charges by Contract and Churn ===")
    print(pivot.round(2))


def engineer_features(df):
    """
    Run full feature engineering pipeline.
    
    Args:
        df (pd.DataFrame): raw DataFrame from load_data
    
    Returns:
        pd.DataFrame: DataFrame with engineered features added
    """
    df=add_charge_per_tenure(df)
    df=add_risk_category(df)
    df = add_is_fully_loaded(df)
    print_pivot_summary(df)
    
    return df