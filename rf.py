# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:58:02 2016

Stats 154 Final Project without stemming

@author: yingyingli
"""

from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
import nltk
import pandas as pd
import re
import string
from string import digits
try:
    maketrans = ''.maketrans
except AttributeError:
    from string import maketrans
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
import enchant

# read data in as data frame
st = pd.read_csv('/Users/yingyingli/Desktop/HRC_train.tsv', sep = 'str("") | str("\t")',  names=['content'], engine='python')
df = pd.DataFrame(st.content.str.split('"\t"',1).tolist(), columns = ['label','content'])

# label
for i in df.index:
    df.label[i] = df.label[i][1]


# get rid of punctuations and numbers
replace_punctuation = maketrans(string.punctuation, ' '*len(string.punctuation))
translator_2 = {ord(k): None for k in digits}
df['upd_content'] = df.content
df['content_non_punc'] = df.content
for i in df.index:
	df['upd_content'][i] = df.content[i].translate(replace_punctuation)
	df['content_non_punc'][i] = df['upd_content'][i].translate(translator_2)

# tokenize all content
df['token1'] = df['content_non_punc'].apply(nltk.word_tokenize)
df['token'] = df['content'].apply(nltk.word_tokenize)


# create a dictionary in the form {word1:count1,...}
dictionary = df.token.apply(Counter)
dictionary1 = df.token1.apply(Counter)
# print(dictionary)

label = np.array(df.label)
# convert the dictionary into a data frame
ve = DictVectorizer(sparse = False)
matrix = ve.fit_transform(dictionary)
#data = np.append(matrix,label.reshape(3505,1),axis = 1)
#names.append('label')
names = ve.get_feature_names()
data = pd.DataFrame(matrix,columns=names)
ve1 = DictVectorizer(sparse = False)
matrix1 = ve1.fit_transform(dictionary1)
names1 = ve1.get_feature_names()
data1 = pd.DataFrame(matrix1,columns=names1)
# print(data)



# filter out words that appear less than 20 times
colsum = data.apply(sum)
colsum1 = data1.apply(sum)

d = data.loc[:,colsum > 50]
d1 = data1.loc[:,colsum1 > 50]

# power feature
# how many words are in each email
d['number_word'] = data.apply(sum, 1)
d1['number_word'] = data1.apply(sum, 1)
# not very useful

# number of characters in each email
number_char = df.content.apply(len)
d['number_char'] = number_char
# not very useful

# phrase_1 = wave attack
phrase_1 = []
for i in range(3505):
    if data1.loc[i, 'wave'] > 0 and data1.loc[i, 'attack'] > 0:
        phrase_1.append(1)
    else:
        phrase_1.append(0)
d['phrase_1'] = phrase_1
# not very useful

# phrase 2 "benghazi", "libya"
phrase_2 = []
for i in range(3505):
    if data1.loc[i, 'benghazi'] > 0 and data1.loc[i, 'libya'] > 0:
        phrase_2.append(1)
    else:
        phrase_2.append(0)
d['phrase_2'] = phrase_2
# sort of useful

# phrase 3 "ambassador", "steven"
phrase_3 = []
for i in range(3505):
    if data1.loc[i, 'ambassador'] > 0 and data1.loc[i, 'steven'] > 0:
        phrase_3.append(1)
    else:
        phrase_3.append(0)
d['phrase_3'] = phrase_3
# not useful


d = data.loc[:,colsum > 50]
# phrase 4: "president", "obama"
phrase_4 = []
for i in range(3505):
    if data1.loc[i, 'president'] > 0 and data1.loc[i, 'obama'] > 0:
        phrase_4.append(1)
    else:
        phrase_4.append(0)
d['phrase_4'] = phrase_4

# phrase 5: "prime", "minister","netanyahu"
phrase_5 = []
for i in range(3505):
    if data1.loc[i, 'president'] > 0 and data1.loc[i, 'obama'] > 0 and data1.loc[i, 'president'] >0:
        phrase_5.append(1)
    else:
        phrase_5.append(0)
d['phrase_5'] = phrase_5


matrix = pd.DataFrame.as_matrix(d)
# random forest
RF = RandomForestClassifier(n_estimators=1000,criterion='entropy',max_features=300,max_depth=100,oob_score=True)
RF.fit(matrix,label)

# print(RF.feature_importances_)
print(RF.oob_score_)







#data.to_csv('~/desktop/data.csv', index=False, header=True, sep=',')
# do not enchant
'''
d = enchant.Dict("en_US")
word_bool = []
words=[]
#punc = []
for i in range(len(names)):
    word_bool.append(d.check(names[i]))
    if d.check(names[i]):
        words.append(names[i])
    #if names[i] in string.punctuation:
        #punc.append(names[i])
        
data1= data.loc[:,words]
#punctuation = data.loc[:,punc]

matrix = pd.DataFrame.as_matrix(data1)
'''
matrix1 = pd.DataFrame.as_matrix(d1)

#master_df = pd.concat([data1, punctuation], axis = 1)
# data1['biaoqian'] = label
#master_df.to_csv('/Users/yingyingli/Desktop/data_with_punc.csv', index=False, header=True, sep=',')
