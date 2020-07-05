# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:57:57 2020

@author: kiran
"""
import numpy as np
import pandas as pd
import string
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

def load_doc(filename):
        file = open(filename,'r')
        text = file.read()
        file.close()
        return text
    
def load_file(file_name):
    dataset = pd.read_csv(file_name)
    return dataset

dataset = load_file('train.csv')

def clean_data(data):
    tokens = data.split()
    #Remove Punchuation
    table = str.maketrans('', '',string.punctuation)
    tokens = [w.translate(w) for w in tokens]
    #Remove Non Charachter 
    tokens = [w for w in tokens if w.isalpha()]
    #Remove Stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    #Remove short words
    tokens = [w for w in tokens if len(w)>1]
    tokens = [w for w in tokens.lower()]
    return tokens
  
    
data = dataset.iloc[:,3]
tokens = clean_data(data[0])

def add_tweet_to_vocab(data,vocab):
    token = clean_data(data)
    vocab.update(token)

def process_all_tweets(data,vocab):
    for i in range(len(data)):
        add_tweet_to_vocab(data[i],vocab)
        
#data = dataset.iloc[:,3]
#target = dataset.iloc[:,4]
#data,target = clean_data(dataset)

vocab = Counter()
process_all_tweets(data,vocab)

min_occurence = 2
tokens =[k for k,c in vocab.items() if c>=min_occurence]
print(len(tokens))

def save_list(lines,filename):
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()
   
    
save_list(tokens,'vocab.txt')

def doc_to_tweet(tweet,vocab):
    tokens = clean_data(tweet)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_all_tweets(data,vocab):
    tweets = []
    for i in range(len(data)):
        tweet = doc_to_tweet(data[i], vocab)
        tweets.append(tweet)
    return tweets

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

tweets = process_all_tweets(data, vocab)


tokenizer =Tokenizer()
tokenizer.fit_on_texts(tweets)

Xtrain = tokenizer.texts_to_matrix(tweets,mode='tfidf')

ytrain = dataset.iloc[:,4]

n_words = Xtrain.shape[1]

model = Sequential()
model.add(Embedding(6168, 200,input_length=4681))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(Xtrain,ytrain,verbose=2,epochs=50)


def predict_sentiment(review,vocab,tokenizer,model):
    token = clean_data(review)
    token = [w for w in token if w in vocab]
    line = ' '.join(token)
    save_test(line,'clean_test3.txt')
    input = tokenizer.texts_to_matrix([line],mode='tfidf')
    ypredict = model.predict(input,verbose=0)
    return round(ypredict[0,0])

def save_test(lines,filename):
    file = open(filename,'a')
    file.write(lines+'\n')
    file.close()
    
def predict_all_test_data():
    data_set = load_file('test.csv')
    xtest = data_set.iloc[:,3]
    predict = list()
    id_ = list()
    all_tweet=list()
    for i in range(len(xtest)):
        yhat = predict_sentiment(xtest[i],vocab, tokenizer, model)
        predict.append(yhat)
        id_.append(data_set.iloc[i,0])
        all_tweet.append(data_set.iloc[i,3])
    return np.asarray(id_),np.asarray(predict),all_tweet
 
"""
id_,predict,all_tweet = predict_all_test_data()
dict = {'id': id_,'tweet':all_tweet ,'target': predict}  
tweet_predict = pd.DataFrame(dict)
tweet_predict.to_csv('tweets2.csv')
"""