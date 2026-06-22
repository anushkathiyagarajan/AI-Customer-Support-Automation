import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/processed/encoded_tickets.csv")

# Handle missing values
df["clean_body"] = df["clean_body"].fillna("")

# Features and Labels
X = df["clean_body"]
y = df["queue_encoded"]

# Tokenization
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)

X_pad = pad_sequences(
    X_seq,
    maxlen=max_len
)

# One-hot encoding labels
num_classes = len(df["queue_encoded"].unique())

y_cat = to_categorical(
    y,
    num_classes=num_classes
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y_cat,
    test_size=0.2,
    random_state=42
)

# LSTM Model
model = Sequential()

model.add(
    Embedding(
        input_dim=max_words,
        output_dim=128,
        input_length=max_len
    )
)

model.add(
    LSTM(64)
)

model.add(
    Dense(
        num_classes,
        activation="softmax"
    )
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nLSTM Accuracy:", accuracy)

# Save Model
model.save(
    "models/lstm_model.h5"
)

joblib.dump(
    tokenizer,
    "models/tokenizer.pkl"
)

print("\nLSTM Model Saved Successfully!")