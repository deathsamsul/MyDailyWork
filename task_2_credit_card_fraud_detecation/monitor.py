import streamlit as st
import pandas as pd
import sqlite3
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff


# streamlit run task_2_credit_card_fraud_detecation/monitor.py

API_UPDATE_URL = "http://127.0.0.1:8000/update_label"

st.set_page_config(page_title="Fraud Monitoring Dashboard", layout="wide")

st.title(" Fraud Model Monitoring Dashboard")


# Load Data
conn = sqlite3.connect("fraud_monitor.db")

df = pd.read_sql("SELECT * FROM predictions", conn)

df["prediction"] = df["prediction"].astype(int)
df["actual_label"] = pd.to_numeric(df["actual_label"], errors="coerce")

# -----------------------------
# ROW 1
# Prediction Overview | Model Performance
# -----------------------------

col1, col2 = st.columns(2)

with col2:

    st.subheader("Prediction Overview")

    st.metric("Total Predictions", len(df))

    fraud_predictions = df["prediction"].sum()

    st.metric("Fraud Predictions", fraud_predictions)

with col1:

    st.subheader("Model Performance")

    df_valid = df.dropna(subset=["actual_label"])

    if len(df_valid) > 0:

        y_true = df_valid["actual_label"].astype(int)
        y_pred = df_valid["prediction"].astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", f"{acc:.2f}")
        m2.metric("Precision", f"{precision:.2f}")
        m3.metric("Recall", f"{recall:.2f}")
        m4.metric("F1 Score", f"{f1:.2f}")

    else:

        st.warning("No labeled data available yet.")

st.divider()



# Confusion Matrix | Fraud Rate
col3, col4 = st.columns(2)

with col4:

    if len(df_valid) > 0:

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["Pred Legit", "Pred Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            colorscale="Blues"
        )

        st.plotly_chart(fig_cm, use_container_width=True)

with col3:

    if len(df_valid) > 0:

        st.subheader("Fraud Rate")

        fraud_rate = (df_valid["actual_label"].sum() / len(df_valid)) * 100

        st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}%")

st.divider()



# Fraud Probability Distribution
st.subheader(" Fraud Probability Distribution")

fig = px.histogram(
    df,
    x="fraud_probability",
    nbins=20,
    title="Fraud Probability Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()



# Model Drift Monitoring

st.subheader(" Model Drift Monitoring")

df["timestamp"] = pd.to_datetime(df["timestamp"])

drift_df = df.groupby(df["timestamp"].dt.date)["fraud_probability"].mean().reset_index()

fig_drift = px.line(
    drift_df,
    x="timestamp",
    y="fraud_probability",
    title="Average Fraud Probability Over Time"
)

st.plotly_chart(fig_drift, use_container_width=True)

st.divider()


# Update Actual Label
st.subheader("🛠 Update Actual Fraud Label")

st.write(
    "Fraud investigators can update the **true label** after reviewing the transaction."
)

st.dataframe(df)

st.divider()

transaction_id = st.text_input("Transaction ID")

actual_label = st.selectbox(
    "Actual Label",
    [0, 1],
    format_func=lambda x: "Legitimate (0)" if x == 0 else "Fraud (1)"
)

if st.button("Update Label"):

    payload = {
        "transaction_id": transaction_id,
        "actual_label": actual_label
    }

    try:

        response = requests.post(API_UPDATE_URL, json=payload)

        if response.status_code == 200:

            st.success("Label updated successfully!")
            st.experimental_rerun()

        else:

            st.error("Failed to update label.")

    except Exception:

        st.error("API connection failed. Make sure FastAPI server is running.")