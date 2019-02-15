'''
Created on Feb 12, 2019

@author: mingw
'''
import pandas as pd   
import sklearn
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from test.test_threading_local import target
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Reading Data and other garbage
stop_words = set(stopwords.words('english'))
training = pd.read_csv('drugLibTrain_raw.tsv', delimiter = '\t', encoding = 'utf-8')
testing = pd.read_csv('drugLibTest_raw.tsv', delimiter = '\t', encoding = 'utf-8')
stemmer = SnowballStemmer("english")
train_data = training[['rating', 'commentsReview']]
test_data = testing[['rating','commentsReview']]
test_data['commentsReview'] = test_data['commentsReview'].astype(str)
train_data['commentsReview'] = train_data['commentsReview'].astype(str)

# All the preprocessing, in one method
def clean_text(text):
    #Lower-case
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    #Stemming and removing stop words, in one line!
    text = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return text


#Vectorizing
train_count_vect = CountVectorizer(analyzer=clean_text)
test_count_vect = CountVectorizer(analyzer=clean_text)
Train_counts = train_count_vect.fit_transform(train_data['commentsReview'])
Test_counts = test_count_vect.fit_transform(test_data['commentsReview'])
# print(X_counts.shape)
train_feature_names = train_count_vect.get_feature_names()
train_labels= train_data['rating']
train_label_names=[1,2,3,4,5,6,7,8,9,10]
train_features=Train_counts.toarray()

test_feature_names = test_count_vect.get_feature_names()
test_label=test_data['rating']
test_label_names=[1,2,3,4,5,6,7,8,9,10]
test_features=Test_counts.toarray()

#Wow I can't believe I got this far. It's now time to do some Machine Learning classifications.
#Gaussian Naive Bayes
gnb=GaussianNB()
gnb_model = gnb.fit(train_features,train_labels)
gnb_preds = gnb.predict(test_features)
gnb_accuracy = accuracy_score(test_label,gnb_preds)
print("Accuracy",gnb_accuracy)