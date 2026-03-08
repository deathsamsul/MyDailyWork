import tensorflow as tf
import pickle
from .custom_layers import TransformerBlock, TokenAndPositionEmbedding


def load_model_and_tokenizer(model_type="word"):

    if model_type == "word":
        model_path = "task_5_hand_written_text_generation/model/word_model/tinystories_lstm_70k_final.keras"
        tokenizer_path = "task_5_hand_written_text_generation/model/word_model/tokenizer.pkl"

    elif model_type == "mini_gpt":
        model_path = "task_5_hand_written_text_generation/model/mini_gpt/mini_gpt_tinystories_1M.keras"
        tokenizer_path = "task_5_hand_written_text_generation/model/mini_gpt/tokenizer.pkl"

    else:
        raise ValueError("Invalid model type")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "TransformerBlock": TransformerBlock,
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "MultiHeadAttention": tf.keras.layers.MultiHeadAttention
        },
        compile=False
    )

    # automatically detect correct sequence length
    seq_length = model.input_shape[1]

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer, seq_length