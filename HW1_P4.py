# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:14:51 2017

@author: Qi Zhao
"""
from matplotlib import style
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
import os

os.chdir('/Users/ap/Dropbox/2017FALL/EECS E6720BayesianModelforML/HW1/EECS6720-hw1-data')
xtrain = pd.read_csv('X_train.csv', header = None)
ytrain = pd.read_csv('label_train.csv', header = None)
xtest = pd.read_csv('X_test.csv', header = None)
ytest = pd.read_csv('label_test.csv', header = None)
# set all hyperparameters
a = 1
b = 1
e = 1
f = 1
N = ytrain.shape[0]
N0 = np.sum(ytrain[0] == 0)
N1 = np.sum(ytrain[0] == 1)
ystar1_giveny = (e + np.sum(ytrain == 1)) / (N + e + f)
ystar0_giveny = (f + np.sum(ytrain == 0)) / (N + e + f)
col_sum0 = np.sum(xtrain.loc[ytrain.index[ytrain[0] == 0].tolist()], 0)
col_sum1 = np.sum(xtrain.loc[ytrain.index[ytrain[0] == 1].tolist()], 0)

def cal_log_negbin(x, alpha, beta):
    log_p = sp.special.gammaln(x + alpha) - sp.special.gammaln(alpha) - \
    sp.special.gammaln(x + 1) + np.log((beta / (beta + 1)) ** alpha) + \
    np.log((1 / (beta + 1)) ** x)
    return(log_p)
    
def pred_1prob(xstar):
    log_p = 0
    for i, v in xstar.iteritems():
        temp = cal_log_negbin(v, a + col_sum1[i], b + N1)
        log_p = log_p + temp
    p = np.exp(log_p)*ystar1_giveny
    return(p)

def pred_0prob(xstar):
    log_p = 0
    for i, v in xstar.iteritems():
        temp = cal_log_negbin(v, a + col_sum0[i], b + N0)
        log_p = log_p + temp
    p = np.exp(log_p)*ystar0_giveny
    return(p)
    
test0 = xtest.apply(pred_0prob, axis = 1)
test1 = xtest.apply(pred_1prob, axis = 1)
test0 = np.asarray(test0[0].tolist())
test1 = np.asarray(test1[0].tolist())
# There are some emails can not be determined. The reason is the number of several variables
# of them are larger enough to let log(predict_probability) = -inf meaning predict_probability 
# is just 0 for both y = 1 or y = 0
num_undetermined = np.count_nonzero(np.where((test1 == 0) & (test0 == 0)))
pred_y = 1 * (test0 < test1)
# If we can not decide them, we tend to regard them are spam since too much same words in them.
pred_y[np.where((test1 == 0) & (test0 == 0))] = 1
v11 = np.sum([a and b for a, b in zip(ytest[0].values == 0, pred_y == 0)])
v12 = np.sum([a and b for a, b in zip(ytest[0].values == 0, pred_y == 1)])
v21 = np.sum([a and b for a, b in zip(ytest[0].values == 1, pred_y == 0)])
v22 = np.sum([a and b for a, b in zip(ytest[0].values == 1, pred_y == 1)])
table = pd.DataFrame({'real_notspam':[v11, v12], 'real_spam':[v21, v22]}, 
                      index = ['predict_notspam', 'predict_spam'])
#### (c) ####
# Pick three mislabeled emails firstly
mis3 = ytest.index[ytest[0] != pred_y].tolist()
mis3 = mis3[:3]
temp = test0 + test1  # temp is the sum of both probabilities
temp[np.where(temp == 0)] = np.float('nan') # Normalize the probabilities, if temo = 0, set it to nan
test0 = np.divide(test0, temp)
test1 = np.divide(test1, temp)
# E(lambda1) = E(E(lambda1|xi:yi=1)), where lambda1|xi:yi=1 is the posterior of
# The lambda1 given the data, specificly, Gamma(1+sum(xi:yi=1), 1+N1), same for lambda0
Elambda1 = (col_sum1 + 1) / (N1 + 1)
Elambda0 = (col_sum0 + 1) / (N0 + 1)
with open('README.txt', 'r') as file:
    xnames = file.read().split('\n')
    
def make_plots(index):
    style.use('ggplot')    
    plt.xticks(range(53), xnames, rotation = 'vertical')
    plt.plot(xtest.loc[index], 'b-', label = 'Features')
    plt.plot(Elambda0, 'r-', label = r'$E(\vec{\lambda_{0}})$')
    plt.plot(Elambda1, 'g-', label = r'$E(\vec{\lambda_{1}})$')    
    plt.legend(loc='best')
    plt.title('The Features of {}th Sample VS '.format(index + 1) + \
              r'$E(\vec{\lambda_{1}})$' + '&' r'$E(\vec{\lambda_{0}})$')
    plt.show()
# Predictive Probability    
def get_pred_prob(index):
    print('P({}th Email is not Spam) ='.format(index + 1), test0[index])
    print('P({}th Email is Spam) ='.format(index + 1), test1[index])    
    
for each in mis3:
    make_plots(each)
    get_pred_prob(each)

#### (d) ####

cloest3 = abs(test0 - test1).argsort()[0:3]
for each in cloest3:
    make_plots(each)
    get_pred_prob(each)
