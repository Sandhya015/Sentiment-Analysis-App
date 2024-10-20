from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from PIL import Image  # Ensure this import works now
import pytesseract


app = Flask(__name__)

# Load the trained LSTM model
model = load_model('sentiment_lstm_model.h5')

# Load the IMDB word index
from tensorflow.keras.datasets import imdb
word_index = imdb.get_word_index()

# Preprocess input text (convert to integer sequences and pad)
def preprocess_text(text):
    words = text.lower().split()
    sequences = []
    
    for word in words:
        if word in word_index and word_index[word] < 10000:  # Consider only top 10,000 words
            sequences.append(word_index[word])
    
    return pad_sequences([sequences], maxlen=100)

# Predict sentiment from text
def predict_sentiment(text):
    padded_input = preprocess_text(text)
    prediction = model.predict(padded_input)

    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = ""

    # Check if image or text input
    if 'image' in request.files and request.files['image'].filename != '':
        image = request.files['image']
        img = Image.open(image)
        text = pytesseract.image_to_string(img)  # Convert image to text
    elif 'text' in request.form and request.form['text'] != '':
        text = request.form['text']
    else:
        return render_template('index.html', sentiment="No input provided", input_text=text)

    # Predict sentiment
    sentiment = predict_sentiment(text)
    
    return render_template('index.html', sentiment=sentiment, input_text=text)

if __name__ == '__main__':
    app.run(debug=True)
