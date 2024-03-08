import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import json
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
    test_path = 'sample-test.in.json'
    with open(test_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]

    # Membuat DataFrame dari data yang sudah dipecah
    test_df = pd.json_normalize(data)

    # Menghapus baris pertama
    test_df = test_df.drop(0, axis=0)
    
    test_df['text'] = test_df['city'] + " " + test_df['section'] + " " + test_df['heading']
    del test_df['city']
    del test_df['section']
    del test_df['heading']

    # Display test data
    st.subheader('Test Data:')

    # Allow user to select a row for prediction
    selected_row = st.selectbox('Select a row for prediction:', test_df['text'])

    # Classify the selected text
    prediction = classify_text(selected_row)

    # Display prediction
    st.subheader('Prediction:')
    st.write({'text': selected_row, 'category': prediction})

if __name__ == '__main__':
    main()
