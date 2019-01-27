'''
Created on Jan 27, 2019

@author: mingw
'''
import sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

iris = load_iris()

label_names = iris.target_names
labels = iris.target
feature_names = iris.feature_names
features = iris.data

train, test, train_labels, test_labels =train_test_split(features, labels, test_size = 0.33, random_state = 42)

gnb = GaussianNB()

model = gnb.fit(train, train_labels)

preds = gnb.predict(test)
print(preds)
score = f1_score(test_labels,preds, average= 'macro')

print(score)
# filename = 'iris.csv'
# data = np.recfromcsv(filename)
# print("Iris")
