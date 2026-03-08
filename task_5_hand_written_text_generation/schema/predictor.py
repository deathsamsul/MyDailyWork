import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences






# GPT MODEL PREDICTION
def predict_top_k_gpt(model, tokenizer, text, seq_length, k=5, temperature=1.0):

    text = text.lower().strip()
    if not text:
        return []

    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-seq_length:]

    token_list = pad_sequences(
        [token_list],
        maxlen=seq_length,
        padding="pre"
    )

    predictions = model.predict(token_list, verbose=0)

    logits = predictions[0, -1]   # GPT last token logits

    logits = logits / temperature
    probs = tf.nn.softmax(logits).numpy()

    top_k_indices = np.argsort(probs)[-k:][::-1]

    results = []
    for idx in top_k_indices:
        word = tokenizer.index_word.get(idx, "")
        probability = float(probs[idx])
        results.append((word, probability))

    return results


# LSTM MODEL PREDICTION


def predict_top_k_lstm(model, tokenizer, text, seq_length=30, k=5, temperature=0.8):
    text = text.lower()
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences(
        [token_list], maxlen=seq_length, padding="pre"
    )
    preds = model.predict(token_list, verbose=0)[0]
    preds = np.log(preds + 1e-9) / temperature
    preds = np.exp(preds)
    preds = preds / np.sum(preds)
    top_idx = preds.argsort()[-k:][::-1]
    results = [(tokenizer.index_word.get(idx, ""), float(preds[idx])) for idx in top_idx]
    return results






# def predict_top_k_lstm(model, tokenizer, text, seq_length, k=5, temperature=1.0):

#     text = text.lower().strip()
#     if not text:
#         return []

#     token_list = tokenizer.texts_to_sequences([text])[0]

#     token_list = pad_sequences(
#         [token_list],
#         maxlen=seq_length,
#         padding='pre'
#     )

#     predictions = model.predict(token_list, verbose=0)[0]  # already probs

#     # Temperature scaling (better way for probabilities)
#     predictions = np.log(predictions + 1e-10) / temperature
#     predictions = np.exp(predictions)
#     predictions = predictions / np.sum(predictions)

    # top_indices = predictions.argsort()[-k:][::-1]

    # results = []
    # for idx in top_indices:
    #     word = tokenizer.index_word.get(idx, "")
    #     probability = float(predictions[idx])
    #     results.append((word, probability))

    # return results


#auto ruter function to call the right predictor based on model type
def predict_top_k(model, tokenizer, text, seq_length, model_type, k=5, temperature=1.0):

    if model_type == "mini_gpt":
        return predict_top_k_gpt(model, tokenizer, text, seq_length, k, temperature)

    elif model_type == "word":
        return predict_top_k_lstm(model, tokenizer, text, seq_length, k, temperature)

    else:
        return []