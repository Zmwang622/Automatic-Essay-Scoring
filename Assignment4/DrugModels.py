'''
Created on Feb 12, 2019

@author: mingw
'''
import pandas as pd   
import sklearn
import nltk
import re
import string
import numpy as np

import matplotlib.pyplot as plt
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
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from nltk.classify.megam import numpy
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

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
count_vect = CountVectorizer(ngram_range=(1,2),analyzer=clean_text, max_features=1)
Train_counts = count_vect.fit_transform(train_data['commentsReview'])
test_counts = count_vect.transform(test_data['commentsReview'])
# print(X_counts.shape)
train_feature_names = count_vect.get_feature_names()
train_labels= train_data['rating']
label_names=[1,2,3,4,5,6,7,8,9,10]

print(Train_counts.shape)
print(test_counts.shape)

test_feature_names = count_vect.get_feature_names()
test_label=test_data['rating']


# Feature Extraction
# selector = SelectKBest(score_func=chi2, k=7000)
# Train_counts= selector.fit_transform(Train_counts,train_labels)
# test_counts= selector.transform(test_counts)
# print(Train_counts.shape)
# print(test_counts.shape)

# test_label_names=[1,2,3,4,5,6,7,8,9,10]
train_features=Train_counts.toarray()
test_features=test_counts.toarray()

# WE TRYNA DO SENTIMENTAL ANALYSIS RN?????
# yeah...this wasnt it chief
# sent_vectorizer = CountVectorizer(analyzer="word", max_features=1,lowercase=False)
# train_sent = sent_vectorizer.fit_transform(train_data['commentsReview'])
# test_sent = sent_vectorizer.transform(test_data['commentsReview'])
sia = SIA()
train_sent = []
 
for text in train_data['commentsReview']:
    pol_score = sia.polarity_scores(text)
    train_sent.append(pol_score['compound'])
     
test_sent = []
for text in test_data['commentsReview']:
    pol_score = sia.polarity_scores(text)
    test_sent.append(pol_score['compound'])
# print(train_sent)
train_sent = np.array(train_sent)

test_sent = np.array(test_sent)
 
sent_train_features = train_sent
sent_test_features = test_sent
 
print(sent_train_features.shape)
print(sent_test_features.shape)
train_features=np.concatenate((train_features,sent_train_features),1)
test_features=np.concatenate((test_features,sent_test_features),1)

#Wow I can't believe I got this far. It's now time to do some Machine Learning classifications.
#Gaussian Naive Bayes
gnb=GaussianNB()
gnb_model = gnb.fit(train_features,train_labels)
gnb_preds = gnb.predict(test_features)
gnb_accuracy = accuracy_score(test_label,gnb_preds)
gnb_kappa = cohen_kappa_score(test_label,gnb_preds)
gnb_qkappa = cohen_kappa_score(test_label,gnb_preds, weights="quadratic")

lr = LogisticRegression(solver = 'lbfgs',multi_class="multinomial")
lr_model = lr.fit(train_features,train_labels)
lr_preds= lr.predict(test_features)
lr_acc =accuracy_score(test_label, lr_preds)
lr_kappa = cohen_kappa_score(test_label,lr_preds)
lr_qkappa = cohen_kappa_score(test_label,lr_preds, weights="quadratic")

rfc = RandomForestClassifier(n_estimators=100, max_depth = 2, random_state=42)
rfc_model = rfc.fit(train_features, train_labels)
rfc_preds = rfc.predict(test_features)
rfc_acc =accuracy_score(test_label, rfc_preds)
rfc_kappa = cohen_kappa_score(test_label,rfc_preds)
rfc_qkappa = cohen_kappa_score(test_label,rfc_preds, weights="quadratic")

svc = SVC(gamma="auto")
svc_model = svc.fit(train_features, train_labels)
svc_preds = svc.predict(test_features)
svc_acc = accuracy_score(test_label,svc_preds)
svc_kappa = cohen_kappa_score(test_label,svc_preds)
svc_qkappa = cohen_kappa_score(test_label,svc_preds, weights="quadratic")

knc = KNeighborsClassifier(n_neighbors=10)
knc_model = knc.fit(train_features, train_labels)
knc_preds = knc.predict(test_features)
knc_acc = accuracy_score(test_label, knc_preds)
knc_kappa = cohen_kappa_score(test_label,knc_preds)
knc_qkappa = cohen_kappa_score(test_label,knc_preds, weights="quadratic")

"""
(q)
 _                           
| |                          
| | ____ _ _ __  _ __   __ _ 
| |/ / _` | '_ \| '_ \ / _` |
|   < (_| | |_) | |_) | (_| |
|_|\_\__,_| .__/| .__/ \__,_|
          | |   | |          
          |_|   |_|   
          
░░░░░░░░░
░░░░▄▀▀▀▀▀█▀▄▄▄▄░░░░
░░▄▀▒▓▒▓▓▒▓▒▒▓▒▓▀▄░░
▄▀▒▒▓▒▓▒▒▓▒▓▒▓▓▒▒▓█░
█▓▒▓▒▓▒▓▓▓░░░░░░▓▓█░
█▓▓▓▓▓▒▓▒░░░░░░░░▓█░
▓▓▓▓▓▒░░░░░░░░░░░░█░
▓▓▓▓░░░░▄▄▄▄░░░▄█▄▀░
░▀▄▓░░▒▀▓▓▒▒░░█▓▒▒░░
▀▄░░░░░░░░░░░░▀▄▒▒█░
░▀░▀░░░░░▒▒▀▄▄▒▀▒▒█░
░░▀░░░░░░▒▄▄▒▄▄▄▒▒█░
 ░░░▀▄▄▒▒░░░░▀▀▒▒▄▀░░
░░░░░▀█▄▒▒░░░░▒▄▀░░░
░░░░░░░░▀▀█▄▄▄▄▀
"""


objects = ('gnb','lr', 'rfc', 'svc', 'knc')

N = 3

all_acc = (gnb_accuracy, lr_acc, rfc_acc, svc_acc, knc_acc)
all_kappa = (gnb_kappa, lr_kappa, rfc_kappa, svc_kappa, knc_kappa)
all_qkappa = (gnb_qkappa, lr_qkappa, rfc_qkappa, svc_qkappa, knc_qkappa)

gnb_measurables= (gnb_accuracy,gnb_kappa,gnb_qkappa)
lr_measurables = (lr_acc, lr_kappa, lr_qkappa)
rfc_measurables = (rfc_acc, rfc_kappa, rfc_qkappa)
svc_measurables = (svc_acc, svc_kappa, svc_qkappa)
knc_measurables = (knc_acc, knc_kappa, knc_qkappa)

fig,ax = plt.subplots()

ind = np.arange(N)
width = 0.15

p1 = ax.bar(ind, gnb_measurables, width, color='r', bottom=0)
p2 = ax.bar(ind+width, lr_measurables, width, color = 'y', bottom = 0)
p3 = ax.bar(ind+(2*width), rfc_measurables, width, color = 'g', bottom = 0)
p4 = ax.bar(ind+(3*width),svc_measurables, width, color = 'b', bottom = 0)
p5 = ax.bar(ind+(4*width), knc_measurables, width, color = 'm', bottom = 0)

ax.set_title('Comparison of 5 Machine Learning Models on Drug Review Dataset')
ax.set_xticks(ind+width/2)
# why does python think kappa is not a word D':
ax.set_xticklabels(('Accuracy','Kappa','Quadratic Kappa'))
ax.legend((p1[0], p2[0],p3[0],p4[0],p5[0]), ('Gaussian Naive-Bayes','Logistic Regression','Random Forest'
                                             ,'Support Vector Classification','k-neighbors Classifier '))
plt.ylabel('Percentage')
ax.set_ybound()
plt.show()