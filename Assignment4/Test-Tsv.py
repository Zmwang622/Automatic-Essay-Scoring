'''
Created on Feb 10, 2019

@author: mingw
'''
import pandas as pd   
import sklearn
import nltk

from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from nltk.sentiment import sentiment_analyzer
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer

stopwords = nltk.corpus.stopwords.words('english')
training = pd.read_csv('drugLibTrain_raw.tsv', delimiter = '\t', encoding = 'utf-8')
testing = pd.read_csv('drugLibTest_raw.tsv', delimiter = '\t', encoding = 'utf-8')
columns = training.columns

print(training.columns)
