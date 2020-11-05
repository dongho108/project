import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
np.random.seed(seed=0)


#data load
data = pd.read_csv("Reviews.csv", nrows = 100000)
data = data[['Text', 'Summary']]

#data refine
data.drop_duplicates(subset=['Text'], inplace=True)
data.dropna(axis=0, inplace=True)

#nltk stop word
nltk.download()
stop_words = set(stopwords.words('english'))
print('불용어 개수 :', len(stop_words))
print(stop_words)


