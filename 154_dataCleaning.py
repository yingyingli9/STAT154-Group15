# -*- coding: utf-8 -*-
"""
Statistics 154 Final Project 

Created on Mon Nov 14 18:13:48 2016

@author: yingyingli
"""

import nltk
import pandas as pd
import re
from collections import Counter
import string

# read data in as data frame
st = pd.read_csv("D:/大学/学习/大三/ML/final/HRC_train.tsv", sep = 'str("") | str("\t")',  names=['content'], engine='python')
df = pd.DataFrame(st.content.str.split('"\t"',1).tolist(), columns = ['label','content'])

# label
for i in df.index:
    df.label[i] = df.label[i][1]
print(df.content)

'''
for j in [x for x in range(3505) if x not in [24,44,81,100,119,132,137,175,193,195,214,215,225,236,237,241]]:
    df.content[j] = re.search(r'subject: (.*?) (u.s. department of state case no.)', df.content[j]).group(1)
'''


# get rid of punctuations and numbers
translator = str.maketrans({key: None for key in string.punctuation})
df['content_non_punc'] = df.content
for i in df.index:
    df['content_non_punc'][i] = df.content[i].translate(translator)

# tokenize all content
df['token'] = df.content_non_punc.apply(nltk.word_tokenize)

# steming correction


# create a dictionary in the form {word1:count1,...}
dictionary = df.token.apply(Counter)


# convert the dictionary into a data frame
names = Counter()
for i in df.index:
    names += dictionary[i]

# analyze data frame to find power feature
