import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the best model from the .pkl file
best_model = joblib.load('best_model.pkl')
nltk.download('punkt')  # Download the punkt tokenizer
nltk.download('stopwords')  # Download the NLTK stop words
stop_words = set(stopwords.words('english'))

# Function to preprocess tokens in a text
def process_tokens(tokens):
    processed_tokens = [re.sub(r'[^a-zA-Z0-9]', '', token.lower()) for token in tokens if token.lower() not in stop_words]
    processed_tokens = list(filter(None, processed_tokens))
    return processed_tokens

# Function to classify a text using the best model
def classify_text(text):
    tokens = word_tokenize(text)
    processed_tokens = process_tokens(tokens)
    tfidf_vectorizer = TfidfVectorizer()
    text_vectorized = tfidf_vectorizer.transform([' '.join(processed_tokens)])
    prediction = best_model.predict(text_vectorized)
    return prediction[0]

# Streamlit App
def main():
    st.title('Text Classification App')

    # Load test data
    test_data_path = 'sample-test.in.json'
    test_data = pd.read_json(test_data_path, lines=True)

    # Display test data
    st.subheader('Test Data:')
    st.write(test_data)

    # Classify each text in the test data
    predictions = []
    for index, row in test_data.iterrows():
        text = row['text']
        prediction = classify_text(text)
        predictions.append({'text': text, 'category': prediction})

    # Display predictions
    st.subheader('Predictions:')
    st.write(predictions)

if __name__ == '__main__':
    main()