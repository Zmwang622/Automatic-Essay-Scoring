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

training = pd.read_csv('drugLibTrain_raw.tsv', delimiter = '\t', encoding = 'utf-8')
testing = pd.read_csv('drugLibTest_raw.tsv', delimiter = '\t', encoding = 'utf-8')

training['commentsReview'] = training['commentsReview'].astype(str)
raw_comments=""
for index, row in training.iterrows():
    raw_comments+= (row['commentsReview'])

tokenized_comments = word_tokenize(raw_comments)

stop_words = set(stopwords.words('english')) #About 150 stopwords

filtered_comments = []
for comment in tokenized_comments:
    if comment not in stop_words:
        filtered_comments.append(comment)

stemmed_comments = []
stemmer = SnowballStemmer("english")
for comment in filtered_comments:
    stemmed_comments.append(stemmer.stem(comment))


    