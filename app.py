from flask import Flask, request, jsonify
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask.helpers import _endpoint_from_view_func
from flask_restplus import Api, Resource, fields
import pandas as pd
import numpy as np
import joblib
import re
import demoji
import itertools
import nltk
from importlib import reload
import logistic_regression
reload(logistic_regression)
from logistic_regression import LogisticRegression 
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
import math


app=Flask(__name__)
api = Api(app, version='0.13.0', title='Sentiment Analysis API', description='API for Sentiment Analysis')
ns = api.namespace('sentiment', description='Sentiment analysis operations')

#Load model
model = LogisticRegression()

#fungsi untuk menghapus mention
def remove_pattern(input_txt, pattern):
  r = re.findall(pattern, input_txt)
  for i in r:
    input_txt = re.sub(i, '', input_txt)
  return input_txt

#fungsi untuk menghapus emoji
# Mengunduh konfigurasi emoji pertama kali
#demoji.download_codes()
def remove_emoji(text):
    return demoji.replace(text, '')

#Translate emoticon
emoticon_df = pd.read_csv('D:/Users/user/Documents/FRESH GRADUATED/Atmatech/DEPLOYMENT/emoticon.txt')
emoticon_dict = dict(zip(emoticon_df[0], emoticon_df[1]))
def translate_emoticon(t):
    for w, v in emoticon_dict.items():
        pattern = re.compile(re.escape(w))
        match = re.search(pattern,t)
        if match:
            t = re.sub(pattern,v,t)
    return t

#Fungsi untuk menghapus format dari kaskus
def remove_kaskus_formatting(text):
    text = re.sub(r'\[', ' [', text)
    text = re.sub(r'\]', '] ', text)
    text = re.sub(r'\[quote[^ ]*\].*?\[\/quote\]', ' ', text)
    text = re.sub(r'\[[^ ]*\]', ' ', text)
    text = re.sub('&quot;', ' ', text)
    return text

#Fungsi untuk menghapus karakter & lowercase
def clean_caracter(text):
    text = re.sub(r'\n', '', text)
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(http[s]?://\S+)", " ", text).split())
    return text.lower()

#Slang Words
slang_words = pd.read_csv('D:/Users/user/Documents/FRESH GRADUATED/Atmatech/DEPLOYMENT/slangword.csv')
slang_dict = dict(zip(slang_words['original'],slang_words['translated']))
def transform_slang_words(text):
    word_list = text.split()
    word_list_len = len(word_list)
    transformed_word_list = []
    i = 0
    while i < word_list_len:
        if (i + 1) < word_list_len:
            two_words = ' '.join(word_list[i:i+2])
            if two_words in slang_dict:
                transformed_word_list.append(slang_dict[two_words])
                i += 2
                continue
        transformed_word_list.append(slang_dict.get(word_list[i], word_list[i]))
        i += 1
    return ' '.join(transformed_word_list)

#Fungsi untuk menghapus URL text
def remove_url(text):
    return re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)

#Fungsi untuk menghapus spasi berlebihan
def remove_whitespace(tweet):
  return ' '. join(tweet.split())

#Fungsi untuk menghapus non-Alphabet
def remove_non_alphabet(text):
    output = re.sub('[^a-zA-Z ]+', '', text)
    return output

#Fungsi untuk menghapus karakter berulang
def remove_repeating_characters(text):
    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))

#Fungsi untuk tokenizing
def tokenizing(tweet):
    tokens = word_tokenize(tweet)
    return ' '.join(tokens)

#Stemming
factory = StemmerFactory() # Membuat objek stemmer
stemmer = factory.create_stemmer()

def stemmed_wrapper(term):
  return stemmer.stem(term)

# Melakukan stemming pada semua token di kolom 'original_text'
def stem_all_tokens(dataframe):
    term_base = {}
    for document in dataframe['original_text']:
        for term in document.split():
            if term not in term_base:
                term_base[term] = " "

    for term in term_base:
        term_base[term] = stemmed_wrapper(term)

    dataframe['original_text'] = dataframe['original_text'].apply(lambda x: ' '.join([term_base[word] for word in x.split()]))

    return dataframe

#Tf-Idf
# Menghitung Term Frequency (TF) untuk kolom 'original_text'
def calculate_tf(document):
    words = document.split()
    word_count = len(words)
    term_frequency = Counter(words)
    tf = {word: count / word_count for word, count in term_frequency.items()}
    return tf

# Menghitung Inverse Document Frequency (IDF) untuk kolom 'original_text'
def calculate_idf(documents):
    word_to_document_count = {}
    total_documents = len(documents)

    for document in documents:
        words = set(document.split())
        for word in words:
            word_to_document_count[word] = word_to_document_count.get(word, 0) + 1

    idf = {word: math.log(total_documents / count) for word, count in word_to_document_count.items()}
    return idf

# Menghitung TF-IDF untuk kolom 'original_text'
def calculate_tfidf(document, idf):
    tf = calculate_tf(document)
    tfidf = {word: tf[word] * idf[word] for word in tf}
    # Mengembalikan hanya bobot TF-IDF sebagai daftar
    return list(tfidf.values())

#Menggabungkan semua langkah preprocessing
def preprocess_text(text, idf):
    text = remove_pattern(text, r"@[\w]*")
    text = remove_emoji(text)
    text = translate_emoticon(text)
    text = remove_kaskus_formatting(text)
    text = clean_caracter(text)
    text = transform_slang_words(text)
    text = remove_url(text)
    text = remove_whitespace(text)
    text = remove_non_alphabet(text)
    text = remove_repeating_characters(text)
    text = tokenizing(text)
    text = calculate_tf(text)
    text = calculate_idf(text)
    tfidf = calculate_tfidf(text, idf)
    return tfidf
max_features = 5000

# Definisi endpoint pertama
@api.route('/label')
class SentimentAnalysis(Resource):
    @api.expect(api.model('Text', {'text': 'Text to be predicted'}))
    def post(self):
        text = request.json['text']

        # Lakukan preprocessing terlebih dahulu
        preprocessed_text = preprocess_text(text)
        processed_text = np.zeros((len(preprocessed_text), max_features))
        for i, tfidf_list in enumerate(preprocessed_text):
            processed_text[i, :len(tfidf_list)] = tfidf_list[:max_features]
        # Selanjutnya, lakukan prediksi pada teks yang sudah diproses
        sentiment = model.predict([processed_text])

        response = {}

        if sentiment < 0.5:
            response['sentiment'] = sentiment[0]
            response['prediction_text'] = 'Ini bukan pencemaran nama baik :)'
        else:
            response['sentiment'] = sentiment[0]
            response['prediction_text'] = 'Ini adalah pencemaran nama baik!!!'

        return jsonify(response)

# Definisi endpoint kedua
@api.route('/label/batch')
class BatchSentimentAnalysis(Resource):
    @api.expect(api.model('BatchText', {'texts': 'List of texts to be predicted'}))
    def post(self):
        texts = request.json['texts']

        # Lakukan preprocessing untuk setiap teks dalam batch
        preprocessed_texts = [preprocess_text(text) for text in texts]

        processed_texts = np.zeros((len(preprocessed_texts), max_features))
        for i, tfidf_list in enumerate(preprocessed_texts):
            processed_texts[i, :len(tfidf_list)] = tfidf_list[:max_features]

        # Selanjutnya, lakukan prediksi pada teks yang sudah diproses
        sentiments = model.predict(processed_texts)

        response = []

        for sentiment in sentiments:
            prediction_text = 'Ini adalah pencemaran nama baik!!!' if sentiment >= 0.5 else 'Ini bukan pencemaran nama baik :)'
            response.append({
                'sentiment': sentiment,
                'prediction_text': prediction_text
            })

        return jsonify(response)



if __name__ == "__main__":
    app.run(debug=True)