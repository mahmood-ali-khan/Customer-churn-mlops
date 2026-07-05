import pandas as pd
from sklearn.preprocessing import StandardScaler

#filepath=r"C:\Users\USER\Documents\AI-portfolio\project 1\customer-churn-mlops\data\churn.xlsx"
def load_data(filepath):
    """ Loading Raw Churn data and droping unnacessary columns
    Args:
       filepath(str): path to churn dataset .xlsx file
    Returns:
       pd.DataFrame: cleaned DataFrame with unnecessary columns removed 
    """
    df=pd.read_excel(filepath)

    col_to_drop=[ 'CustomerID', 'Count', 'Country', 'State', 'City',
    'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
    'Churn Reason', 'Churn Score', 'Churn Value', 'Total Charges']

    df=df.drop(columns=col_to_drop)
    return df

def fix_dtypes(df):
    """Correcting the data type in DataFrames
    
    Args:
       df(pd.dataFrame):cleaned dataframe from load_data
    returns:
        pd.dataframe: with correct data types
    """
    df['Churn Label']=(df['Churn Label']=='Yes').astype(int)
    df=df.rename(columns={'Churn Label':'churn'})
    return df

def encode_features(df):
    """
    Encode categorical features for machine learning.
    
    Args:
        df (pd.DataFrame): DataFrame from fix_dtypes
    
    Returns:
        pd.DataFrame: DataFrame with encoded features
    """
    binary_cols = [
     'Senior Citizen', 'Partner', 'Dependents',
    'Phone Service', 'Paperless Billing']
    multi_cols = [
    'Multiple Lines', 'Internet Service', 'Online Security',
    'Online Backup', 'Device Protection', 'Tech Support',
    'Streaming TV', 'Streaming Movies', 'Contract', 'Payment Method']
    
    #For Binary Feature Encoding
    df['Gender']=(df['Gender']=='Male').astype(int)
    
    for col in binary_cols:
        df[col]=(df[col]=='Yes').astype(int)


    #For multi-category Feature Encoding
    df= pd.get_dummies(df, columns=multi_cols,drop_first=True, dtype=int)

    return df


def scale_features(df):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df (pd.DataFrame): encoded DataFrame from encode_features
    
    Returns:
        pd.DataFrame: DataFrame with scaled numerical features
    """
    num_cols=['Tenure Months', 'Monthly Charges', 'CLTV']
    scaler=StandardScaler()
    df[num_cols]=scaler.fit_transform(df[num_cols])

    return df

def preprocess(filepath):
    """
    Full preprocessing pipeline.
    
    Args:
        filepath (str): path to churn.xlsx file
    
    Returns:
        pd.DataFrame: fully preprocessed DataFrame ready for modelling
    """
    df=load_data(filepath)
    df=fix_dtypes(df)
    df=encode_features(df)
    df=scale_features(df)

    return df
