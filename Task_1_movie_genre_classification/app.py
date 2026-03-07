import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle
import os
import joblib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shema import data_preprocessing





# streamlit run task_1_movie_genre_classification/app.py

st.set_page_config(page_title="Movie Genre Classifier", layout="wide")


#streamlit run movie_genre_prediction/TFID_vector/main.py

st.markdown("""
###  About This Model

This is a heigly imbalanced  multi-class classification model trained on **27 different movie genres**.
Because the dataset is highly imbalanced and some genres have similar themes,
the model may not always predict the exact genre correctly.
Instead of accuracy, we evaluate performance using **Macro F1-score**,
which is more suitable for imbalanced multi-class problems.""")
st.divider()
with st.expander("About me"):
    st.write("Hi, I am Samsul — Machine Learning, Deep Learning, NLP & GenAI Enthusiast.")

comparison_df = pd.DataFrame({
    "Model": ["TF-IDF", "Fine-Tune : DeBERTa"],
    "Speed": ["Fast ", "Slower "],
    "Accuracy": ["Medium", "High"],
    "GPU Required": ["No", "Optional"]
})

st.subheader("Model Comparison")
st.table(comparison_df)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_deberta_model():

    model_path = os.path.join(BASE_DIR, "model", "distilbert_model", "deberta_movie_genre_model_gpu_v1")
    label_path = os.path.join(BASE_DIR, "model", "distilbert_model", "label_encoder_movie_genre_model_gpu_v1.pkl")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    label_encoder = joblib.load(label_path)

    model.to(device)
    return model, tokenizer, label_encoder


# Load model
@st.cache_resource
def load_tfidf_model():

    model_path = os.path.join(BASE_DIR, "model", "TF_IDF_model", "movie_27_genre_classifier.pkl")
    label_path = os.path.join(BASE_DIR, "model", "TF_IDF_model", "label_encoder.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(label_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, label_encoder



#wweb ui
st.title(" Movie Genre Classification")
st.write("Predict movie genre using title and summary")
st.divider()
st.subheader("Enter Movie Details")


model_choice = st.selectbox(
    "Choose Model",
    ["TF-IDF (Fast)", "DeBERTa (High Accuracy)"]
)

# Input fields (because your model uses title + summary)
title = st.text_input("Movie Title")
summary = st.text_area("Movie Summary", height=200)


#prediction
if st.button("Predict Genre"):

    if title.strip() == "" or summary.strip() == "":
        st.warning("Please enter both title and summary.")
    else:
        input_df = pd.DataFrame({
            "title": [title],
            "summary": [summary]
        })

        # Preprocess input for TF-IDF model
        if model_choice == "TF-IDF (Fast)":
            model, label_encoder = load_tfidf_model()

            probabilities = model.predict_proba(input_df)[0]
            genre_names = label_encoder.classes_
        

        # Preprocess input for DeBERTa model
        else:
            model, tokenizer, label_encoder = load_deberta_model()

            text = title + " " + summary

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=192
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]

            probabilities = probs.cpu().numpy()
            genre_names = label_encoder.classes_

        #output results
        results_df = pd.DataFrame({
            "Genre": genre_names,
            "Probability": probabilities
        })




        results_df = results_df.sort_values(by="Probability", ascending=False)

        top_genre = results_df.iloc[0]

        st.success(f"Top Predicted Genre: **{top_genre['Genre']}**")
        st.info(f"Confidence: {top_genre['Probability']*100:.2f}%")

        st.subheader("Probability Distribution")
        st.bar_chart(results_df.set_index("Genre"))

        st.subheader("Top 3 Genre Predictions")
        st.table(results_df.head(3))

        st.subheader("All Genre Probabilities")
        st.dataframe(results_df)

       

st.divider()

st.markdown("""
##  About This Model

This is a highly imbalanced multi-class classification model trained on **27 movie genres**.

### Models Used:

**1️.TF-IDF + Logistic Regression**
- Fast inference
- Lightweight
- CPU-friendly

**2️. DeBERTa Transformer Model**
- Fine-tuned on movie plot summaries
- Higher accuracy
- Uses GPU if available

Evaluation Metric:
- **Macro F1-score** (better for imbalanced datasets)
""")