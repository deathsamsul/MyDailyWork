import pandas as pd

# same service columns used during training
services = [
    'PhoneService','MultipleLines','InternetService','OnlineSecurity',
    'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]


def feature_engineering(df):

    # avoid division by zero
    df['avg_spend'] = df['TotalCharges'] / (df['tenure'] + 1)

    df['loyalty_label'] = pd.cut(
        df['tenure'],bins=[0,12,24,48,72],labels=[1,2,3,4])

    df['services'] = df[services].isin(["Yes","DSL","Fiber optic"] ).sum(axis=1)

    df['per_cost'] = df['MonthlyCharges'] / (df['services'] + 1)

    df['charges_yr'] = df['MonthlyCharges'] * 12 / (df['tenure'] + 1)

    return df