# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:49:38 2020

@author: GRAFFJE
"""

import nltk
import pandas as pd
import numpy as np
import sklearn
import re  

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns

## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import os
import os.path

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import gensim
from pprint import pprint
from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel

sns.set_style('whitegrid')

path = r'C:\Users\Jpuwi\Documents\Syracuse_University\Spring2020\IST-736\Group_Project\master_speech_latest.xlsx'
tempfile = pd.ExcelFile(path)
rawfile = tempfile.parse(0)
print(rawfile)

# Collapsing a bunch of columns into one column
rawfile['comb_text'] = rawfile[rawfile.columns[19:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)), axis = 1)

# 0 = democrat, 1 = republican
df_party = rawfile.copy(deep=True)

# Below we can select the 3 columns we want instead of
# specifying a bunch of columns we don't want
df_party = df_party.filter(['id', 'party_num', 'comb_text'])
# df_party.drop(['candidate','party','year','incumbent_candidate','incumbent_party','winner','text','text2','text3','text4','text5'],axis = 1, inplace = True)
print(df_party)

PartyList = []
TextList = []
IndexList = []

# https://www.geeksforgeeks.org/create-a-list-from-rows-in-pandas-dataframe/
for row in df_party.itertuples():
    partylabel = row.party_num
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("\'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    textlabel = textlabel.replace('nan',' ') # remove nan

    indexlabel = row.id
    TextList.append(textlabel)
    PartyList.append(partylabel)
    IndexList.append(indexlabel)

print(IndexList)
print(PartyList)
print(TextList)

# Instantiating a stemmer object
stemmer = PorterStemmer()
# defining and creating a stemmer function
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [stemmer.stem(word) for word in words]
    return words
############# Now - use CountVectorizer.....................

MyVect = CountVectorizer(input = 'content',
                         ngram_range = (3, 3),
                         stop_words = 'english',
                         max_features = 300,
                         token_pattern = '(?u)[a-zA-Z]+[a-zA-Z{1}]')

## Now I can vectorize using my list of complete paths to my files
X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
CorpusDF_text=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(CorpusDF_text.head())

# LDA Part starts here
# -------- LDA -------
# Instantiating an LDA object
LDA_model = LatentDirichletAllocation(n_components = 4, max_iter = 600,
                                      learning_method = 'online')
# Fitting the LDA model
LDA_vec = LDA_model.fit_transform(X_text)

# Let's see how the first document in the corpus look like
# in different topic space
print(LDA_vec[0])

# implement a topic printing function
def print_topics(model, vectorizer, top_n = 10):
    for idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % (idx))
        #print([(vectorizer.get_feature_names()[i])
        #print([(vectorizer.get_feature_names()[i], topic[i])
        print(", ".join([vectorizer.get_feature_names()[i]
              for i in topic.argsort()[:-top_n - 1:-1]]))
               #for i in topic.argsort()[: -top_n - 1:-1]])
               
# printing words and their topics               
print_topics(LDA_model, MyVect, 100)

# Visualizations
# --------------
import pyLDAvis.sklearn as LDAvis
import pyLDAvis

panel = LDAvis.prepare(LDA_model, X_text, MyVect, mds = 'tsne')
pyLDAvis.show(panel)

# define and create a function to print top 10 most common words
# Source: https://bit.ly/36JzQJN
def plot_10_most_common_words(X_text, MyVect):
    import matplotlib.pyplot as plt
    words = MyVect.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in X_text:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize = (15, 15/1.6180))
    plt.subplot(title = '10 most common phrases')
    sns.set_context("notebook", font_scale = 2, rc = {"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation = 90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

# Plot the top ten words by frequency
plot_10_most_common_words(X_text, MyVect)

# Word Cloud
## Start of word cloud
from wordcloud import WordCloud
# Join the corpus text together
long_string = ','.join(list(ColumnNamesText))
# create a WordCloud object
wordcloud = WordCloud(background_color = "white", max_words = 100, 
                      contour_width = 3, width = 800, height = 400, 
                      contour_color = "steelblue")
# Generate a wordcloud
wordcloud.generate(long_string)
wordcloud.to_image()
