#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:22:36 2018

@author: subramanianiyer
"""

import pickle
import pandas as pd
import numpy as np
from copy import deepcopy
import sys
from keras.models import Sequential
from keras.layers import Dense, Embedding, Reshape, Activation, SimpleRNN, GRU, LSTM, Convolution1D, MaxPooling1D, Merge, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical
from IPython.display import SVG
from keras.preprocessing import sequence
#from keras.utils.vis_utils import model_to_dot, plot_model
import keras
# x is real, y is predicted
def Accuracy(x,y):
    return float(sum([int(a==b) for a,b in zip(x,y)]))/float(len(x))
def Precision(x,y):
    #Everythin you got right as a 1 divided by everything you said was 1
    return float(sum([int(a==b and b==1) for a,b in zip(x,y)])/float(sum([int(b==1) for b in y])))
def Recall(x,y):
    #Everything you got right as a 1 divided by everything that actually was 1
    return float(sum([int(a==b) and a==1 for a,b in zip(x,y)])/float(sum([int(a==1) for a in x])))
forewards = [[1,5,10,20,30,45,60,90,120],[210,270,330,390,450,510,600,720,960]]
backshift = 960
LSTMwidth = np.round(2*backshift, decimals = -3)
thresh = np.linspace(0,.3,num=3001)
with open('check3x1.pkl', 'rb') as picklefile:
    data = pickle.load(picklefile)
def discify(x,X):
    for i in range(1,x+1):
        X['target'] = [z+y for z,y in zip(X['target'], X['logC'].shift(-1*i))]
company = 8 #can be 0 through 9
fsec = 0 # can be 0 or 1
for stock in range(company, company+1):
    balances = []
    predBalances = []
    threshes = []
    returns = []
    accuracies = []
    precisions = []
    noInvsts = []
    for foreward in forewards[fsec]:
        test = deepcopy(data[stock])
        X = deepcopy(test)
        X['target'] = X['logC']
        discify(foreward,X)
        for i in range(1,backshift+1):
            X['logC'+str(i)] = X['logC'].shift(i)
        del X['logC']
        del X['TradedVolume']
        X.dropna(inplace = True)
        lc = deepcopy(X['target'])
        del X['target']
        y = [x>0 for x in lc]
        a = len(X)
        s = int(a*.7)
        trainX = X[:s]
        trainy = y[:s]
        test0X = X[s:]
        test0y = y[s:]
        trainX2 = np.asarray(trainX).reshape((trainX.shape[0], 1, trainX.shape[1]))
        test0X = np.asarray(test0X).reshape((test0X.shape[0], 1, test0X.shape[1]))
        keras.backend.clear_session()
        monkey = Sequential()
        monkey.add(LSTM(LSTMwidth,input_shape = (trainX2.shape[1], trainX2.shape[2])))
        monkey.add(Dense(2))
        monkey.add(Activation('softmax'))
        monkey.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
        monkey.fit(trainX2,np.asarray(trainy),nb_epoch=30, validation_data=(test0X,np.asarray(test0y)))
        balances.append(1 - sum(test0y)/len(test0y))
        preds = monkey.predict(test0X)
        predBalances.append(float(sum(x[0]>x[1] for x in preds))/float(len(preds)))
        rets = []
        del monkey
        for i in thresh:
            hm = [int(x[1]-x[0]>i) for x in preds]
            cum = np.zeros(foreward)
            tc = list(lc[s:])
            for j in range(len(tc)):
                if hm[j]==1:
                    cum[j%foreward]+=tc[j]
            rr = np.exp(cum)
            rets.append(np.average(rr))
        returns.append(np.max(np.asarray(rets)))
        threshes.append(thresh[np.argmax(np.asarray(rets))])
        hm = [int(x[1]-x[0]>threshes[-1]) for x in preds]
        tt = [int(x) for x in test0y]
        accuracies.append(Accuracy(tt, hm))
        precisions.append(Precision(tt, hm))
        print('foreward '),
        print(foreward),
        print(' done.')
    results = [balances, predBalances, threshes, returns, accuracies, precisions, noInvsts]
    with open('stock' + str(stock) + str(fsec) + '.pkl','wb') as pfile:
        pickle.dump(results, pfile)
print('stock complete')
