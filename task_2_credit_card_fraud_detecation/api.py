from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sqlite3
import uuid
#from predictor import predict_fraud
from task_2_credit_card_fraud_detecation.predictor import predict_fraud



# uvicorn task_2_credit_card_fraud_detecation.api:app --reload


# Initialize FastAPI app object
app = FastAPI(title="Fraud Detection API")

# Database connection
conn = sqlite3.connect("fraud_monitor.db", check_same_thread=False)

# Create table
conn.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    transaction_id TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    fraud_probability REAL,
    prediction INTEGER,
    actual_label INTEGER
)
""")

from pydantic import BaseModel

class Transaction(BaseModel):

    merchant: str = "fraud_Rippin, Kub and Mann"
    category: str = "misc_net"
    amt: float = 100.0
    gender: str = "M"
    city: str = "Altona"
    state: str = "NY"
    zip: int = 12910
    lat: float = 44.8865
    long: float = -73.5766
    city_pop: int = 4304
    job: str = "Furniture designer"
    unix_time: int = 1371816865
    merch_lat: float = 44.959148
    merch_long: float = -73.6224
    trans_date_trans_time: str = "2013-06-21 12:14:25"
    dob: str = "1968-03-19"


@app.post("/predict")

def predict(transaction: Transaction):

    input_data = transaction.dict()

    pred, prob = predict_fraud(input_data)

    transaction_id = str(uuid.uuid4())

    conn.execute("""
    INSERT INTO predictions(transaction_id,fraud_probability,prediction)
    VALUES(?,?,?)
    """,(transaction_id, prob, pred))

    conn.commit()

    return {
        "transaction_id":transaction_id,
        "fraud_probability":prob,
        "prediction":pred
    }




# After fraud investigation, we store the true label.
class LabelUpdate(BaseModel):
    transaction_id:str
    actual_label:int


@app.post("/update_label")

def update_label(data:LabelUpdate):

    conn.execute("""
    UPDATE predictions
    SET actual_label=?
    WHERE transaction_id=?
    """,(data.actual_label,data.transaction_id))

    conn.commit()

    return {"message":"Label Updated"}