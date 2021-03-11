#!/usr/bin/env python
# coding: utf-8

# In[187]:


import nltk
import pandas as pd
import numpy as np
import sklearn
import re
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

import os
import os.path

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string


# In[188]:


tempfile = pd.ExcelFile("C:/Users/David Yamin/.spyder-py3/master_speech.xlsx")
rawfile = tempfile.parse(0)
print(rawfile)
rawfile['comb_text'] = rawfile[rawfile.columns[19:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)), axis = 1)


# In[189]:


df_party = rawfile.copy(deep=True)
df_party = df_party.filter(['id', 'party_num', 'comb_text'])
print(df_party)

PartyList = []
TextList = []
IndexList = []


# In[190]:


for row in df_party.itertuples():
    partylabel = row.party_num
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("\'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    PartyList.append(partylabel)
    IndexList.append(indexlabel)


print(IndexList)
print(PartyList)
print(TextList)


# In[191]:


## next label
## winner
# 1 = won presidency, 0 = did not win
df_winner = rawfile.copy(deep=True)
df_winner = df_winner.filter(['id', 'winner', 'comb_text'])
print(df_winner)

WinnerList = []


for row in df_winner.itertuples():
    winnerlabel = row.winner
    WinnerList.append(winnerlabel)
    
print(WinnerList)

## next label
## sentiment
# 1 = positive, 0 = negative
df_sent = rawfile.copy(deep=True)
df_sent = df_sent.filter(['id', 'sentiment', 'comb_text'])
print(df_sent)

SentimentList = []

for row in df_sent.itertuples():
    sentimentlabel = row.sentiment
    SentimentList.append(sentimentlabel)
    
print(SentimentList)

## next label
## incumbent_candidate
# 1 = positive, 0 = negative
df_incum_cand = rawfile.copy(deep=True)
df_incum_cand = df_incum_cand.filter(['id', 'incumbent_candidate', 'comb_text'])
print(df_incum_cand)

IncumCandList = []

for row in df_incum_cand.itertuples():
    incum_candlabel = row.incumbent_candidate
    IncumCandList.append(incum_candlabel)
    
print(IncumCandList)

## next label
## incumbent_party
# 1 = positive, 0 = negative
df_incum_party = rawfile.copy(deep=True)
df_incum_party = df_incum_party.filter(['id', 'incumbent_party', 'comb_text'])
print(df_incum_party)

IncumPartyList = []

for row in df_incum_party.itertuples():
    incum_partylabel = row.incumbent_party
    IncumPartyList.append(incum_partylabel)
    
print(IncumPartyList)

## next label
## unemployment
# 1 = positive, 0 = negative
df_unemployment = rawfile.copy(deep=True)
df_unemployment = df_unemployment.filter(['id', 'unemployment', 'comb_text'])
print(df_unemployment)

UnemploymentList = []

for row in df_unemployment.itertuples():
    unemploymentlabel = row.unemployment
    UnemploymentList.append(unemploymentlabel)
    
print(UnemploymentList)

## next label
## GDP
# 1 = positive, 0 = negative
df_GDP = rawfile.copy(deep=True)
df_GDP = df_GDP.filter(['id', 'GDP', 'comb_text'])
print(df_GDP)

GDPList = []

for row in df_GDP.itertuples():
    GDPlabel = row.GDP
    GDPList.append(GDPlabel)
    
print(GDPList)

## next label
## Inflation
# 1 = positive, 0 = negative
df_Inflation = rawfile.copy(deep=True)
df_Inflation = df_Inflation.filter(['id', 'Inflation', 'comb_text'])
print(df_Inflation)

InflationList = []

for row in df_Inflation.itertuples():
    Inflationlabel = row.Inflation
    InflationList.append(Inflationlabel)
    
print(InflationList)

## next label
## satisfaction
# 1 = positive, 0 = negative
df_satisfaction = rawfile.copy(deep=True)
df_satisfaction = df_satisfaction.filter(['id', 'satisfaction', 'comb_text'])
df_satisfaction = df_satisfaction[df_satisfaction['satisfaction'].notna()]
print(df_satisfaction)

SatisfactionList = []

for row in df_satisfaction.itertuples():
    satisfactionlabel = row.satisfaction
    SatisfactionList.append(satisfactionlabel)

SatisfactionList = [ int(x) for x in SatisfactionList ]
print(SatisfactionList)

## next label
## real_income_growth
# 1 = positive, 0 = negative
df_real_inc_growth = rawfile.copy(deep=True)
df_real_inc_growth = df_real_inc_growth.filter(['id', 'real_income_growth', 'comb_text'])
df_real_inc_growth = df_real_inc_growth[df_real_inc_growth['real_income_growth'].notna()]
print(df_real_inc_growth)

RealIncomeGrowthList = []

for row in df_real_inc_growth.itertuples():
    real_inc_growthlabel = row.real_income_growth
    RealIncomeGrowthList.append(real_inc_growthlabel)

RealIncomeGrowthList = [ int(x) for x in RealIncomeGrowthList ]
print(RealIncomeGrowthList)

## next label
## pres_approval
# 1 = positive, 0 = negative
df_pres_approval = rawfile.copy(deep=True)
df_pres_approval = df_pres_approval.filter(['id', 'pres_approval', 'comb_text'])
df_pres_approval = df_pres_approval[df_pres_approval['pres_approval'].notna()]
print(df_pres_approval)

PresApprovalList = []

for row in df_pres_approval.itertuples():
    pres_approvallabel = row.pres_approval
    PresApprovalList.append(pres_approvallabel)

PresApprovalList = [ int(x) for x in PresApprovalList ]
print(PresApprovalList)

## next label
## income_tax
# 1 = positive, 0 = negative
df_income_tax = rawfile.copy(deep=True)
df_income_tax = df_income_tax.filter(['id', 'income_tax', 'comb_text'])
df_income_tax = df_income_tax[df_income_tax['income_tax'].notna()]
print(df_income_tax)

IncomeTaxList = []

for row in df_income_tax.itertuples():
    income_taxlabel = row.income_tax
    IncomeTaxList.append(income_taxlabel)

IncomeTaxList = [ int(x) for x in IncomeTaxList ]
print(IncomeTaxList)

## next label
## djia_volume
# 1 = positive, 0 = negative
df_djia_volume = rawfile.copy(deep=True)
df_djia_volume = df_djia_volume.filter(['id', 'djia_volume', 'comb_text'])
df_djia_volume = df_djia_volume[df_djia_volume['djia_volume'].notna()]
print(df_djia_volume)

DjiaVolumeList = []

for row in df_djia_volume.itertuples():
    djia_volumelabel = row.djia_volume
    DjiaVolumeList.append(djia_volumelabel)

DjiaVolumeList = [ int(x) for x in DjiaVolumeList ]
print(DjiaVolumeList)

## next label
## cpi
# 1 = positive, 0 = negative
df_cpi = rawfile.copy(deep=True)
df_cpi = df_cpi.filter(['id', 'cpi', 'comb_text'])
df_cpi = df_cpi[df_cpi['cpi'].notna()]
print(df_cpi)

CpiList = []

for row in df_cpi.itertuples():
    cpilabel = row.cpi
    CpiList.append(cpilabel)

CpiList = [ int(x) for x in CpiList ]
print(CpiList)

## next label
## cci_index
# 1 = positive, 0 = negative
df_cci_index = rawfile.copy(deep=True)
df_cci_index = df_cci_index.filter(['id', 'cci_index', 'comb_text'])
df_cci_index = df_cci_index[df_cci_index['cci_index'].notna()]
print(df_cci_index)

CciIndexList = []

for row in df_cci_index.itertuples():
    cci_indexlabel = row.cci_index
    CciIndexList.append(cci_indexlabel)

CciIndexList = [ int(x) for x in CciIndexList ]
print(CciIndexList)


#######################################################################################
### all of the Lists in one spot for building out labeled dataframes for model building

print(IndexList)
print(TextList)
print(PartyList)
print(WinnerList)
print(SentimentList)
print(IncumCandList)
print(IncumPartyList)
print(UnemploymentList)
print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)


# In[ ]:





# In[192]:


print(WinnerList)


# In[193]:


df = pd.DataFrame(data = TextList)


# In[194]:


df2 = pd.DataFrame(data = WinnerList)


# In[195]:


frames = [df2, df]
train = pd.concat([df2, df], axis=1).reindex(df.index)


# In[196]:


print(train)


# In[197]:


def createID(r1, r2): 
    return list(range(r1, r2+1)) 
      
# Driver Code 
r1, r2 = 0, 54
id = (createID(r1, r2)) 


# In[198]:


label = {'ids': id, 'labels': WinnerList}
print(label)


# In[199]:


texts = {'ids': id, 'text': TextList}


# In[200]:


df1 = pd.DataFrame(label, columns = ['ids', 'labels'])
df2 = pd.DataFrame(texts, columns = ['ids', 'text'])
train = pd.merge(df1, df2, on='ids', how='inner')
train = train.drop(columns = 'ids', axis =1)
print(train)


# In[201]:


y=train['labels'].values
X=train['text'].values
print(train.head(15))


# In[202]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train[0])
print(y_train[0])
print(X_test[0])
print(y_test[0])


# In[203]:


# Check how many training examples in each category\n",
# this is important to see whether the data set is balanced or skewed
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)))


# In[204]:


#Print out the category distribution in the test data set

unique, counts = np.unique(y_test, return_counts=True)
print(np.asarray((unique, counts)))


# In[205]:


#Step 3: Vectorization
# sklearn contains two vectorizers
# CountVectorizer can give you Boolean or TF vectors
# TfidfVectorizer can give you TF or TFIDF vectors
#Read the sklearn documentation to understand all vectorization options

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
sw = ['responsibilities', 'property', 'sick','show', 'much', 'something', 'mr','years', 'like','put', 'twenty', 'moved', 'would', 'year', 'told', 'single', 'party', 'position', 'well', 'gone', 'went', 'parties', 'weeks', 'going', 'shall','without', 'goes', 'months', 'seven', 'truman','yet', 'seemed', 'treated', 'expected', 'proud', 'develop', 'body', 'wife', 'thousands', 'expect', 'two', 'reasonable', 'agreements', 'written', 'bring', 'expect', 'walk', 'women', 'wrote', 'agreement', 'tonight', 'seem', 'bless', 'existence', 'millions', 
      'mines', 'senator', 'write', 'let', 'thing', 'giving', 'play', 'platform', 'see', 'working', 'work','workers', 'executive', 'plainly', 'second', 'thus', 'bitter', 'give', 'million', 'word', 'sea','woman','ready', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
print(sw)

# several commonly used vectorizer setting
# unigram boolean vectorizer, set minimum document frequency to 5
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words=sw)
#unigram term frequency vectorizer, set minimum document frequency to 5
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=5, stop_words=sw)
# unigram and bigram term frequency vectorizer, set minimum document frequency to 5
unigram12_count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(1,2), min_df=5, stop_words=sw)
#  unigram tfidf vectorizer, set minimum document frequency to 5
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df=5, stop_words=sw)


# In[206]:


## Step 3.1: Vectorize the training data
# The vectorizer can do \"fit\" and \"transform\"
# fit is a process to collect unique tokens into the vocabulary
# transform is a process to convert each document to vector based on the vocabulary
# These two processes can be done together using fit_transform(), or used individually: fit() or transform()
# fit vocabulary in training documents and transform the training documents into vectors

X_train_vec = unigram_count_vectorizer.fit_transform(X_train)

# check the content of a document vector
print(X_train_vec.shape)
print(X_train_vec[0].toarray())

# check the size of the constructed vocabulary
print(len(unigram_count_vectorizer.vocabulary_))
# print out the first 10 items in the vocabulary
print(list(unigram_count_vectorizer.vocabulary_.items())[:10])
# check word index in vocabulary
print(unigram_count_vectorizer.vocabulary_.get('party'))


# In[207]:


## Step 3.2: Vectorize the test data
# use the vocabulary constructed from the training data to vectorize the test data.
# Therefore, use \"transform\" only, not \"fit_transform\"
# otherwise \"fit\" would generate a new vocabulary from the test data
X_test_vec = unigram_count_vectorizer.fit_transform(X_test)
# print out #examples and #features in the test set
print(X_test_vec.shape)



# In[208]:


# Exercise B
# In the above sample code, the term-frequency vectors were generated for training and test data.
# Some people argue that 
# because the MultinomialNB algorithm is based on word frequency,
# we should not use boolean representation for MultinomialNB
# While in theory it is true, you might see people use boolean representation for MultinomialNB
# especially when the chosen tool, e.g. Weka, does not provide the BernoulliNB algorithm.
# sklearn does provide both MultinomialNB and BernoulliNB algorithms.
# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
# You will practice that later
# In this exercise you will vectorize the training and test data using boolean representation
# You can decide on other options like ngrams, stopwords, etc.

# Your code starts here
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# unigram boolean vectorizer, set minimum document frequency to 5
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words=sw)

## Step 3.1: Vectorize the training data

X_train_vec = unigram_bool_vectorizer.fit_transform(X_train)

# check the content of a document vector
print(X_train_vec.shape)
print(X_train_vec[0].toarray())

# check the size of the constructed vocabulary
print(len(unigram_bool_vectorizer.vocabulary_))
# print out the first 10 items in the vocabulary
print(list(unigram_bool_vectorizer.vocabulary_.items())[:10])
# check word index in vocabulary
print(unigram_bool_vectorizer.vocabulary_.get('Carter'))

## Step 3.2: Vectorize the test data

X_test_vec = unigram_bool_vectorizer.transform(X_test)

# print out #examples and #features in the test set

print(X_test_vec.shape)

# Your code ends here"


# In[209]:


# Step 4: Train a MNB classifier

# import the MNB module
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# initialize the MNB model
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

nb_clf= MultinomialNB()
# use the training data to train the MNB model
nb_clf.fit(X_train_vec,y_train)


# Step 4.1 Interpret a trained MNB model"
## interpreting naive Bayes models
## by consulting the sklearn documentation you can also find out feature_log_prob_, 
## which are the conditional probabilities
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
# -8.98942647599 -> logP('worthless'|very negative')
# -11.1864401922 -> logP('worthless'|negative')
# -12.3637684625 -> logP('worthless'|neutral')
# -11.9886066961 -> logP('worthless'|positive')
# -11.0504454621 -> logP('worthless'|very positive')
# the above output means the word feature \"worthless\" is indicating \"very negative\" 
# because P('worthless'|very negative) is the greatest among all conditional probs
unigram_count_vectorizer.vocabulary_.get('soft')
for i in range(0,1):
    print(nb_clf.feature_log_prob_[i][unigram_count_vectorizer.vocabulary_.get('taxes')])


# In[210]:


# sort the conditional probability for category 0 \"very negative\"\n",
# print the words with highest conditional probs
# these can be words popular in the \"very negative\" category alone, or words popular in all cateogires
feature_ranks = sorted(zip(nb_clf.feature_log_prob_[0], unigram_count_vectorizer.get_feature_names()))
negative_features = feature_ranks[-25:]
print(negative_features)


# In[211]:


# Step 5: Test the MNB classifier
# test the classifier on the test data set, print accuracy score
print(nb_clf.score(X_test_vec,y_test))
# print confusion matrix (row: ground truth; col: prediction)
from sklearn.metrics import confusion_matrix
y_pred = nb_clf.fit(X_train_vec, y_train).predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=[0,1])
print(cm)
# print classification report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
from sklearn.metrics import classification_report
target_names = ['0','1']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[212]:


# Step 5.1 Interpret the prediction result"
## find the calculated posterior probability\n",
posterior_probs = nb_clf.predict_proba(X_test_vec)
print(posterior_probs[0])
# find the category prediction for the first test example
y_pred = nb_clf.predict(X_test_vec)
print(y_pred[0])
# check the actual label for the first test example
print(y_test[0])

#Because the posterior probability for category 2 (neutral) is the greatest, 0.50, the prediction should be \"2\".
#Because the actual label is also \"2\", this is a correct prediction


# In[213]:


# Step 5.2 Error Analysis
# print out specific type of error for further analysis
# print out the very positive examples that are mistakenly predicted as negative
# according to the confusion matrix, there should be 53 such examples
# note if you use a different vectorizer option, your result might be different
err_cnt = 0
for i in range(0, len(y_test)):
    if(y_test[i]==0 and y_pred[i]==1):
       print(X_test[i])
       err_cnt = err_cnt+1
print("errors:", err_cnt)


# In[214]:


# Exercise D
text = [
      "this is the opposite of a truly magical movie",
      "achieves the remarkable feat of squandering a topnotch foursome of actors",
      "a deeply unpleasant experience",
      "hugely overwritten",
      "is not Edward Burns' best film",
      "Once the expectation of laughter has been quashed by whatever obscenity is at hand , even the funniest idea isn't funny.",
      "is a deeply unpleasant experience.",
      "is hugely overwritten,",
      "is the opposite of a truly magical movie.",
      "to this shocking testament to anti-Semitism and neo-fascism",
      "is about as humorous as watching your favorite pet get buried alive"]
      #errors: 11
 # Can you find linguistic patterns in the above errors?
# What kind of very positive examples were mistakenly predicted as negative?
# Can you write code to print out the errors that very negative examples were mistakenly predicted as very positive?
# Can you find lingustic patterns for this kind of errors?
# Based on the above error analysis, what suggestions would you give to improve the current model?\n",
# Your code starts here
err_cnt = 0
for i in range(0, len(y_test)):
    if(y_test[i]==1 and y_pred[i]==0):
       print(X_test[i])
       err_cnt = err_cnt+1
print("errors:", err_cnt)
# Your code ends here"
#errors: 30


# In[215]:


# cross validation - Multinomial NB with TF vectors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

nb_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=False, stop_words = sw)),
                        ('nb', MultinomialNB(alpha=.028))])
                                                                                                                         
scores = cross_val_score(nb_clf_pipe, X, y, cv=8)
avg=sum(scores)/len(scores)
print("Multinomial NB average score:", round(avg,3))

#confusion matrix
y_pred = cross_val_predict(nb_clf_pipe, X, y, cv=8)##########
print(confusion_matrix(y, y_pred))

#precision/recall scores (TPs (1s))
print("Precision Score TPs:", round(precision_score(y, y_pred),3)) 

print("Recall Score TPs:", round(recall_score(y, y_pred),3))

#precision/recall scores (TNs (0s))

print("Precision Score TNs:", round(11/16,3))
print("Recall Score TNs:", round(11/26,3))


f = f1_score(y, y_pred)
print("F-1 Score:",round(f,3))


# In[ ]:





# In[36]:


#Bernoulli
nb_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=True, stop_words = sw)),
                        ('bern', BernoulliNB())])
scores = cross_val_score(nb_clf_pipe, X, y, cv=9)
avg=sum(scores)/len(scores)
print("Bernoulli NB average score:", round(avg,4))

#confusion matrix
y_pred = cross_val_predict(nb_clf_pipe, X, y, cv=9)##########
print(confusion_matrix(y, y_pred))

#precision/recall scores (TPs (1s))
print("Precision Score TPs:", round(precision_score(y, y_pred),3)) 

print("Recall Score TPs:", round(recall_score(y, y_pred),3))

#precision/recall scores (TNs (0s))

print("Precision Score TNs:", round(10/15,3))
print("Recall Score TNs:", round(10/26,3))


f = f1_score(y, y_pred)
print("F-1 Score:",round(f,3))


# In[38]:


# cross validation - Multinomial NB with TF-IDF vectors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
nb_clf_pipe = Pipeline([('vect', TfidfVectorizer(encoding='latin-1',   binary=False, stop_words = sw)),
                        ('nb', MultinomialNB(alpha = .163)
                                                                                                                         )])
scores = cross_val_score(nb_clf_pipe, X, y, cv=5)
avg=sum(scores)/len(scores)
print("Multinomial NB=TFIDF average score:", round(avg,3))

#confusion matrix
y_pred = cross_val_predict(nb_clf_pipe, X, y, cv=5)##########
print(confusion_matrix(y, y_pred))

#precision/recall scores (TPs (1s))
print("Precision Score TPs:", round(precision_score(y, y_pred),3)) 

print("Recall Score TPs:", round(recall_score(y, y_pred),3))

#precision/recall scores (TNs (0s))

print("Precision Score TNs:", round(3/3,3))
print("Recall Score TNs:", round(3/26,3))


f = f1_score(y, y_pred)
print("F-1 Score:",round(f,3))


# ### 

# In[ ]:





