import pandas as pd
import nltk
import numpy as np
from bs4 import BeautifulSoup
import unicodedata
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import re
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
stop_words = set(stopwords.words('english')) 
import pickle
pickle_in = open("tokenizer.pickle","rb")
tokenizer = pickle.load(pickle_in)

model =  load_model('sent.h5')

def preprocess(x):
    ps = PorterStemmer()
    x = str(x).lower()
    x = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x) # remove email
    x = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x) # remove urls
    x = BeautifulSoup(x, 'lxml').get_text().strip() # remove html tag
    x = re.sub(r'\brt\b', '', x).strip() # remove retweets
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore') # Remove Accented Characters
    x = re.sub(r'[^\w ]+', "", x) # Special Characters 
    x = ' '.join([t for t in x.split() if t not in stop_words]) # Remove Stop words
    x = ' '.join(ps.stem(i) for i in x.split()) # Stemming
    return x

st.write('# Sentiment Analysis App')
txt = st.text_area("Enter text")
if st.button('Predict'):
 
    cleaned_text = preprocess(txt)
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(tokenized_text,maxlen=20)
    pred = model.predict(padded)[0]
#     st.write(pred)
    if pred[1] >= 0.6:
        st.write("### Negative")
    else:
        st.write("### Positive")
