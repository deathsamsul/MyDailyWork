import pandas as pd
import numpy as np






# Final Model Feature Order

MODEL_COLUMNS = [
            'merchant','category','amt','gender','city','state','zip','lat',
            'long','city_pop','job','unix_time','merch_lat','merch_long',
            'hour','day','month','weekday','age','distance','amt_log',
            'is_night','is_weekend']


# Feature Engineering
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Convert datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # Time Features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["weekday"] = df["trans_date_trans_time"].dt.weekday

    # Age
    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year

    # Distance
    df["distance"] = np.sqrt((df["lat"] - df["merch_lat"])**2 +(df["long"] - df["merch_long"])**2)

    # Log Amount
    df["amt_log"] = np.log1p(df["amt"])

    # Night Transaction
    df["is_night"] = df["hour"].apply(lambda x: 1 if (x >= 22 or x <= 4) else 0)

    # Weekend Transaction
    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)

    # Drop Raw Columns
    drop_cols = ["Unnamed: 0","first", "last", "street", "cc_num", "trans_num", "trans_date_trans_time", "dob" ]

    df = df.drop(columns=drop_cols, errors="ignore")

    df = df[MODEL_COLUMNS]

    return df