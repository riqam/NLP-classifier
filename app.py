import streamlit as st
import pandas as pd
import nltk
import json
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

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

# Streamlit App
def main():
    st.title('Text Classification App')

    # Load test data
    test_path = 'sample-test.in.json'
    with open(test_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file if line.strip()]

    # Membuat DataFrame dari data yang sudah dipecah
    test_df = pd.json_normalize(data)

    test_df['text'] = test_df['city'] + " " + test_df['section'] + " " + test_df['heading']
    del test_df['city']
    del test_df['section']
    del test_df['heading']

    # Menghapus baris pertama
    test_df = test_df.drop(0, axis=0)
    test_df = test_df.reset_index(drop=True)
    
    test_df['tokens'] = test_df['text'].apply(lambda x: process_tokens(word_tokenize(x)))

    text_vectorized = joblib.load('tfidf_vectorizer.pkl').transform(test_df['tokens'].apply(lambda x: ' '.join(x)))
    
    # Display test data
    st.subheader('Test Data:')

    # Allow user to select a row for prediction
    selected_row = st.selectbox('Select a row for prediction:', test_df['text'])
    
    # Get the index of the selected row
    selected_index = test_df[test_df['text'] == selected_row].index[0]

    # Perform prediction only for the selected row
    selected_prediction = best_model.predict(text_vectorized[selected_index])

    # Display prediction
    st.subheader('Prediction:')

    # Menampilkan hasil prediksi berdasarkan teks yang dipilih
    result_df = pd.DataFrame({'Predicted_Category': selected_prediction})
    st.table(result_df)

if __name__ == '__main__':
    main()
