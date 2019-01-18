'''
Created on Jan 17, 2019

@author: mingw
'''
import nltk;
from nltk.tokenize import word_tokenize, sent_tokenize

import codecs


f = open('bodies.csv', encoding = "utf-8")
bodies_raw = f.read() + '\n'

bodies_tokenized = word_tokenize(bodies_raw)

file = open('Tokenized_Bodies_nltk.txt', 'w', encoding = 'utf-8')
s1='\n'.join(bodies_tokenized)
#    for item in bodies_tokenized:
#        file.write("%s\n" % item)
file.write(s1)
file.close()
print("File Written!")
