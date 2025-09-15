import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# Load data
with open("data/faqs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index)

# Generate input sequences
input_sequences = []
for sentence in text.split("\n"):
    tokenized = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized)):
        input_sequences.append(tokenized[:i+1])

# Pad sequences
max_len = max([len(seq) for seq in input_sequences])
padded = pad_sequences(input_sequences, maxlen=max_len, padding="pre")

X = padded[:, :-1]
y = padded[:, -1]
y = to_categorical(y, num_classes=total_words + 1)

# Build Model
model = Sequential([
    Embedding(total_words + 1, 100, input_length=max_len - 1),
    LSTM(150),
    Dense(total_words + 1, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train
history = model.fit(X, y, epochs=100, verbose=1)

# Save model + tokenizer
os.makedirs("models", exist_ok=True)
model.save("models/next_word_model.h5")

import pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
