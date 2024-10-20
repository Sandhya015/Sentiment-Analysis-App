# sentiment_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the IMDB dataset from TensorFlow
from tensorflow.keras.datasets import imdb

# Set the number of words to keep in the vocabulary (top 10,000 most frequent words)
vocab_size = 10000
max_length = 100  # Max number of words per review

# Load the dataset (it is already preprocessed: reviews are converted to integers)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure all input sequences are the same length
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Define the LSTM model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')  # Binary classification (positive/negative)
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (you can adjust epochs depending on your resources)
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64)

# Save the model
model.save("sentiment_lstm_model.h5")
