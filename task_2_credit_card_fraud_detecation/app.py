import streamlit as st
import pandas as pd
import plotly.express as px
from predictor import predict_fraud







st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Title
st.title("💳 Credit Card Fraud Detection System")
st.write("Analyze transactions to determine whether they are fraudulent.")

st.info("""
### 📌 How to Use This App
1️⃣ Choose a sample transaction or enter your own data  
2️⃣ Fill in the transaction details  
3️⃣ Click **Predict Fraud** to analyze the transaction  

⚠️ Higher fraud probability indicates higher fraud risk.
""")

st.divider()

# Sample Transactions

st.subheader(" Try Sample Transactions")

col1, col2 = st.columns(2)

with col1:
    if st.button("Use Legit Transaction"):
        st.session_state.sample = {
            "merchant": "Amazon",
            "category": "shopping_net",
            "amt": 45.20,
            "gender": "F",
            "city": "New York",
            "state": "NY",
            "zip": 10001,
            "lat": 40.7128,
            "long": -74.0060,
            "city_pop": 8000000,
            "job": "Engineer",
            "unix_time": 1371816917,
            "merch_lat": 40.7306,
            "merch_long": -73.9352,
            "trans_time": "2023-01-01 14:30:00",
            "dob": "1990-01-01"
        }

with col2:
    if st.button("Use Fraud Transaction"):
        st.session_state.sample = {
            "merchant": "fraud_Kirlin and Sons",
            "category": "shopping_net",
            "amt": 2500.75,
            "gender": "M",
            "city": "Los Angeles",
            "state": "CA",
            "zip": 90001,
            "lat": 34.0522,
            "long": -118.2437,
            "city_pop": 4000000,
            "job": "Doctor",
            "unix_time": 1371816917,
            "merch_lat": 36.1699,
            "merch_long": -115.1398,
            "trans_time": "2023-01-01 02:30:00",
            "dob": "1985-05-15"
        }

sample = st.session_state.get("sample", {})

st.divider()


# Input Tabs

tab1, tab2, tab3 = st.tabs([
    "💳 Transaction Info",
    "👤 Customer Info",
    "🏪 Merchant Info"
])

# Transaction Tab

with tab1:

    merchant = st.text_input(
        "Merchant",
        value=sample.get("merchant", "")
    )

    category = st.text_input(
        "Category",
        value=sample.get("category", "")
    )

    amt = st.number_input(
        "Transaction Amount ($)",
        value=float(sample.get("amt", 0.0))
    )

    unix_time = st.number_input(
        "Unix Time",
        value=int(sample.get("unix_time", 0))
    )

    trans_time = st.text_input(
        "Transaction Time (YYYY-MM-DD HH:MM:SS)",
        value=sample.get("trans_time", "")
    )

# Customer Tab


with tab2:

    gender = st.selectbox(
        "Gender",
        ["M", "F"],
        index=0 if sample.get("gender", "M") == "M" else 1
    )

    city = st.text_input(
        "City",
        value=sample.get("city", "")
    )

    state = st.text_input(
        "State",
        value=sample.get("state", "")
    )

    zip_code = st.number_input(
        "ZIP Code",
        value=int(sample.get("zip", 0))
    )

    lat = st.number_input(
        "Customer Latitude",
        value=float(sample.get("lat", 0))
    )

    long = st.number_input(
        "Customer Longitude",
        value=float(sample.get("long", 0))
    )

    city_pop = st.number_input(
        "City Population",
        value=int(sample.get("city_pop", 0))
    )

    job = st.text_input(
        "Customer Job",
        value=sample.get("job", "")
    )

    dob = st.text_input(
        "Date of Birth (YYYY-MM-DD)",
        value=sample.get("dob", "")
    )

# Merchant Tab

with tab3:

    merch_lat = st.number_input(
        "Merchant Latitude",
        value=float(sample.get("merch_lat", 0))
    )

    merch_long = st.number_input(
        "Merchant Longitude",
        value=float(sample.get("merch_long", 0))
    )

st.divider()


# Prediction

if st.button("Predict Fraud"):

    input_data = {
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "gender": gender,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "unix_time": unix_time,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "trans_date_trans_time": trans_time,
        "dob": dob
    }

    pred, prob = predict_fraud(input_data)

    st.subheader("📊 Prediction Result")

    # Fraud Message
    if pred == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"Fraud Probability: **{prob:.2%}**")

    # Risk Level
    if prob < 0.3:
        st.success("🟢 Low Fraud Risk")
    elif prob < 0.7:
        st.warning("🟡 Medium Fraud Risk")
    else:
        st.error("🔴 High Fraud Risk")

    # Donut Chart

    fraud_prob = prob
    legit_prob = 1 - prob

    fig = px.pie(
        names=["Legitimate", "Fraud"],
        values=[legit_prob, fraud_prob],
        hole=0.6
    )

    fig.update_traces(textinfo="percent+label")

    st.plotly_chart(fig)