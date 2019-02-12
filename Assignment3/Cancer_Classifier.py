'''
Created on Jan 27, 2019

@author: mingw
'''

#Cross-Validation ML
import sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from Classifier import gnb_model, gnb_preds

cancer = load_breast_cancer()

label_names = cancer.target_names
labels = cancer.target
feature_names = cancer.feature_names
features = cancer.data

train, test, train_labels, test_labels =train_test_split(features, labels, test_size = 0.33, random_state = 42)

gnb = GaussianNB()
gnb_model= gnb.fit(train, train_labels)
gnb_preds = gnb.predict(test)

gnb_f1_score = f1_score(test_labels,gnb_preds, average= 'macro')
gnb_accuracy = accuracy_score(test_labels, gnb_preds)
print("Achieved a",gnb_f1_score,"F1 score and a",gnb_accuracy,"with the Gaussian NB model.")

lr = LogisticRegression()
lr_model = lr.fit(train,train_labels)
lr_preds = lr.predict(test)

lr_f1_score = f1_score(test_labels,lr_preds, average = 'macro')
lr_accuracy = accuracy_score(test_labels, lr_preds)
print("Achieved a",lr_f1_score,"F1 score and a",lr_accuracy,"accuracy with the Logistic Regression model")
print(lr.coef_)
