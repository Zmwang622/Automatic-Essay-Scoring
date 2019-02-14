'''
Created on Feb 12, 2019

@author: mingw
'''
import pandas as pd   
import sklearn
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
#Reading Data
training = pd.read_csv('drugLibTrain_raw.tsv', delimiter = '\t', encoding = 'utf-8')
testing = pd.read_csv('drugLibTest_raw.tsv', delimiter = '\t', encoding = 'utf-8')

training['commentsReview'] = training['commentsReview'].astype(str)
raw_comments=""
for index, row in training.iterrows():
    raw_comments+= (row['commentsReview'])

#Tokenization
tokenized_comments = word_tokenize(raw_comments)

#Stop-Word Removal
stop_words = set(stopwords.words('english')) #About 150 stopwords

filtered_comments = []
for comment in tokenized_comments:
    if comment not in stop_words:
        filtered_comments.append(comment)

#Stemming
stemmed_comments = []
stemmer = SnowballStemmer("english")
for comment in filtered_comments:
    stemmed_comments.append(stemmer.stem(comment))

#N-gram Processing
unigrams= ngrams(stemmed_comments,1)
bigrams = ngrams(stemmed_comments,2)
print(Counter(bigrams))
# transformer = TfidfTransformer(smooth_idf = False)
# tfidf = transformer.fit_transform(unigrams)
#TF-IDF Vectorizer, Count Vectorizer


