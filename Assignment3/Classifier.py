'''
Created on Jan 27, 2019

@author: mingw
'''

#Cross-Validation ML
import sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
iris = load_iris()

print(iris.data)
label_names = iris.target_names
labels = iris.target
feature_names = iris.feature_names
features = iris.data

train, test, train_labels, test_labels =train_test_split(features, labels, test_size = 0.33, random_state = 42)

gnb = GaussianNB()
gnb_model = gnb.fit(train, train_labels)
gnb_preds = gnb.predict(test)
#print(gnb_preds)
gnb_f1_score = f1_score(test_labels,gnb_preds, average= 'macro')
gnb_accuracy = accuracy_score(test_labels, gnb_preds)
#print(gnb_preds)
#print("Achieved a",gnb_f1_score,"F1 score and a",gnb_accuracy,"with the Gaussian NB model.")
train, test, train_labels, test_labels =train_test_split(features, labels, test_size = 0.33, random_state = 42)
lr = LogisticRegression(solver = 'lbfgs',multi_class="multinomial")

lr_model = lr.fit(train, train_labels)
lr_preds = lr.predict(test)

print(lr.coef_)
#lr_f1_score = f1_score(test_labels,lr_preds, average = 'macro')
#lr_accuracy = accuracy_score(test_labels, lr_preds)
print(lr_model.classes_)
#print(lr_preds)
#print("Achieved a",lr_f1_score,"F1 score and a",lr_accuracy,"accuracy with the Logistic Regression model")
# filename = 'iris.csv'
# data = np.recfromcsv(filename)
# print("Iris")
