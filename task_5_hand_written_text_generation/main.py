import streamlit as st
import pandas as pd
from schema.model_load import load_model_and_tokenizer
from schema.predictor import predict_top_k



# streamlit run task_5_hand_written_text_generation/main.py 


# Page Config
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🤖",
    layout="centered"
)

st.title(" Next Word Prediction App")
st.markdown("See Top-5 predictions with probabilities.")





# Model Selection

model_type = st.selectbox(
    "Select Model",
    ("word", "mini_gpt")
)

@st.cache_resource
def load_selected_model(model_type):
    return load_model_and_tokenizer(model_type)

model, tokenizer, seq_length = load_selected_model(model_type)


# User Input
user_input = st.text_input("Enter your text:")

temperature = st.slider(
    "Temperature (Creativity)",
    min_value=0.5,
    max_value=1.5,
    value=0.9,
    step=0.1
)

k = st.slider(
    "Top-K Predictions",
    min_value=3,
    max_value=10,
    value=5
)

# Prediction Button
if st.button("Show Predictions"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        results = predict_top_k(
                    model,
                    tokenizer,
                    user_input,
                    seq_length,
                    model_type=model_type,
                    k=k,
                    temperature=temperature
            )   

        if not results:
            st.error("No predictions found.")
        else:
            df = pd.DataFrame(results, columns=["Word", "Probability"])
            df["Probability"] = df["Probability"].round(4)

            st.subheader("Top Predictions")
            st.dataframe(df, use_container_width=True)

            st.subheader("Probability Distribution")
            st.bar_chart(df.set_index("Word"))