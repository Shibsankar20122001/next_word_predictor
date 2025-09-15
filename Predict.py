import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model & tokenizer
model = load_model("models/next_word_model.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1] + 1  # same maxlen as training

def predict_next_words(text, n=20):
    for _ in range(n):
        tokenized = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([tokenized], maxlen=max_len-1, padding="pre")
        pos = np.argmax(model.predict(padded, verbose=0))

        for word, index in tokenizer.word_index.items():
            if index == pos:
                text += " " + word
                break
    return text

print(predict_next_words("My name is Shibsankar", 20))
