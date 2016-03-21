# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:14:18 2016

@author: rsk
"""


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from collections import Counter

from LogLayer import *

dataset = "/home/rsk/Documents/MNIST/Misclassification/mnist.pkl.gz"
f = gzip.open(dataset, 'rb')
[(trainx,trainy),(validx,validy),(testx,testy)] = cPickle.load(f)
f.close()

#%%
 #train_set, valid_set, test_set
    
def imbalance(data,label,digits={3:0.5,7:0.7}):
    """
    Creates an imabalance in the data vector by selectively downsampling 
    classes as given by the digits dictionary.
    
    The digits dictionary specifies by what ratio any particular class should be
    downsampled
    
    Returns the downsampled datasets and its corresponding labels
    
    """
    
    classIndices=[]

    for i in range(10):
        classIndices.append([])
        
    for i in range(len(data)):
        classIndices[label[i]].append(i)
        
    finalIndices=[]
    
    for i in range(10):
        
        if i in digits.keys():
            n = int(len(classIndices[i])*digits[i])
            
            finalIndices= finalIndices + classIndices[i][0:n]
            
        else:
            finalIndices = finalIndices + classIndices[i]
            
    return [data[finalIndices] , label[finalIndices]]
    
        
#%%
if __name__=="__main__":        
    a = imbalance(trainx,trainy)  

# Checking if imbalance has been created or not
    
    balanced = Counter(trainy)
    imbalanced = Counter(a[1])
    
    print "Balanced :"
    print balanced
    
    print "Imbalanced :"
    print imbalanced
