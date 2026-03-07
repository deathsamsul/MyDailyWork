import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from schema import feature_engineering
import matplotlib.pyplot as plt

# Load model
model = CatBoostClassifier()
model.load_model("task_3_customer_churn_prediction/model/catboost_churn_model.cbm")

st.title("📊 Customer Churn Prediction System")

st.write("Enter customer information to predict churn risk.")

# Create 3 input columns
col1, col2, col3 = st.columns(3)

# Column 1
with col1:
    gender = st.selectbox("Gender", ["Male","Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["Yes","No"])
    tenure = st.slider("Tenure (Months)",0,72,12)

# Column 2
with col2:
    PhoneService = st.selectbox("Phone Service",["Yes","No"])
    MultipleLines = st.selectbox("Multiple Lines",["Yes","No","No phone service"])
    InternetService = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
    OnlineSecurity = st.selectbox("Online Security",["Yes","No","No internet service"])
    OnlineBackup = st.selectbox("Online Backup",["Yes","No","No internet service"])

# Column 3
with col3:
    DeviceProtection = st.selectbox("Device Protection",["Yes","No","No internet service"])
    TechSupport = st.selectbox("Tech Support",["Yes","No","No internet service"])
    StreamingTV = st.selectbox("Streaming TV",["Yes","No","No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies",["Yes","No","No internet service"])
    
TotalCharges = st.number_input("Total Charges",0.0,10000.0,1000.0)
MonthlyCharges = st.number_input("Monthly Charges",0.0,200.0,70.0)
Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])

PaperlessBilling = st.selectbox("Paperless Billing",["Yes","No"])

PaymentMethod = st.selectbox(
"Payment Method",
["Electronic check","Mailed check",
"Bank transfer (automatic)","Credit card (automatic)"]
)

# Create dataframe
data = pd.DataFrame({
'gender':[gender],
'SeniorCitizen':[SeniorCitizen],
'Partner':[Partner],
'Dependents':[Dependents],
'tenure':[tenure],
'PhoneService':[PhoneService],
'MultipleLines':[MultipleLines],
'InternetService':[InternetService],
'OnlineSecurity':[OnlineSecurity],
'OnlineBackup':[OnlineBackup],
'DeviceProtection':[DeviceProtection],
'TechSupport':[TechSupport],
'StreamingTV':[StreamingTV],
'StreamingMovies':[StreamingMovies],
'Contract':[Contract],
'PaperlessBilling':[PaperlessBilling],
'PaymentMethod':[PaymentMethod],
'MonthlyCharges':[MonthlyCharges],
'TotalCharges':[TotalCharges]
})

# Feature engineering
data = feature_engineering(data)

# Prediction button
if st.button("Predict Churn"):

    prob = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"⚠ Customer likely to churn (Risk: {prob:.2f})")

        st.markdown("### 🎁 Retention Offer Suggestions")

        st.info("""
        To retain this customer, consider offering:

        • 20% discount on next 3 months  
        • Free premium streaming upgrade  
        • Loyalty reward points  
        • Priority customer support  
        """)

    else:
        st.success(f"✅ Customer likely to stay (Confidence: {1-prob:.2f})")

        st.markdown("### ⭐ Customer Engagement Suggestions")

        st.success("""
        This customer is loyal. You can improve engagement by:

        • Offering long-term contract discounts  
        • Recommending premium packages  
        • Providing referral bonuses  
        • Upselling additional services
        """)



    st.write("### Churn Risk Level")

    if prob > 0.7:
        st.error("High Risk Customer")
    elif prob > 0.4:
        st.warning("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")

    # Probability Chart
    # Probability Chart
    st.write("### 📈 Churn Probability Visualization")

    fig, ax = plt.subplots()

    labels = ["Stay", "Churn"]
    values = [1-prob, prob]

    ax.bar(labels, values)
    ax.set_ylabel("Probability")
    ax.set_title("Churn Prediction Probability")
    ax.set_ylim(0,1)

    for i,v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')

    # Display chart in Streamlit
    st.pyplot(fig)