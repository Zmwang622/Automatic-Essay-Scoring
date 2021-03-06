'''
Created on Jan 17, 2019

@author: mingw
'''
import nltk;
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

f = open('bodies.csv', encoding = "utf-8")
bodies_raw = f.read() + '\n'
bodies_tokenized = word_tokenize(bodies_raw)
bodies_tokenized=nltk.pos_tag(bodies_tokenized)
bodies_lemmatized = [];
lemmatizer = WordNetLemmatizer()
for token in bodies_tokenized:
    bodies_lemmatized.append(lemmatizer.lemmatize(token))
file = open('Lemmatized_Bodies_nltk.txt', 'w', encoding = 'utf-8')
s1='\n'.join(bodies_lemmatized)
#    for item in bodies_tokenized:
#        file.write("%s\n" % item)
file.write(s1)
file.close()
print("File Written!")