import pickle
import pandas as pd
import numpy as np
from flask import Flask , request , jsonify , render_template
import string 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import nltk
import os

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()


stopwords = set(ENGLISH_STOP_WORDS)
stopwords.discard('no')
stopwords.discard('not')
stopwords.discard('never')
stopwords.discard('nor')





app = Flask(__name__)



def clean(text):
    import re
    # Lowercase
    text = text.lower()

    # Remove URLs and mentions
    text = re.sub(
        r'https?://\S+|www\.\S+|@\w+',
        '',
        text
    )

    # Remove # symbol
    text = re.sub(r'#', '', text)

    # Remove punctuation
    exclude = string.punctuation
    text = text.translate(
        str.maketrans('', '', exclude)
    )

    # Keep only English letters and whitespace
    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    # Remove stopwords
    text = " ".join(
        word for word in text.split()
        if word not in stopwords
    )

    # Lemmatization
    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
    )

    return text

def clean_series(texts):
    return texts.apply(clean)

with open('Hate_speech_pipeline2.pkl' , 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    text = ''
    prediction = None
    if request.method == 'POST':
        text = request.form.get('text', '')
        text_series = pd.Series([text])
        pred = model.predict(text_series)[0]
        prediction = "Hate speech" if pred == 1 else "No Hate speech"
    
    return render_template('index.html' , prediction = prediction , input_text = text)
@app.route('/predict' , methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'Error': 'Input data is not provided'}), 400
        
        text = data.get('text', '')
        text_series = pd.Series([text])
        prediction = model.predict(text_series)[0]

        response = {
            "Prediction": "Hate speech" if prediction == 1 else 'No Hate speech'
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'Error': str(e)}), 500

if __name__ =='__main__':
    port = int(os.environ.get('PORT' , 5000))
    app.run(host = '0.0.0.0' , port = port , debug = True)
