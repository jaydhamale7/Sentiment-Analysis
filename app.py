import re
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stop_words = set(stopwords.words('english'))  
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

model = load_model('model.h5')
ps = PorterStemmer()
def preprocess(x):
    x = str(x).lower()
    x = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x) # remove email
    x = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x) #urls
    x = BeautifulSoup(x, 'lxml').get_text().strip() # remove html tag
    x = re.sub(r'\brt\b', '', x).strip()
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    x = ' '.join([t for t in x.split() if t not in stop_words])
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(ps.stem(i) for i in x.split())
    return x

st.write('# Sentiment Analysis App')
user_input = st.text_area("Enter Text")

if st.button("Predict"):
    x = preprocess(user_input)
    x = [one_hot(words,3000) for words in [x]]
    x = pad_sequences(x,maxlen=20,)
    pred = np.argmax(model.predict(x))
    if pred == 0:
        st.write(f"## Positive")
    else:
        st.write(f"## Negative")