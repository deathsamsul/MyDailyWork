import pandas as pd
from catboost import CatBoostClassifier
from task_2_credit_card_fraud_detecation.schem import feature_engineering



# Load model
model = CatBoostClassifier()
model.load_model("task_2_credit_card_fraud_detecation/model/credit_fraud_detection.cmb")


def predict_fraud(input_data):

    df = pd.DataFrame([input_data])

    # Feature engineering
    df = feature_engineering(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return int(prediction), float(probability)