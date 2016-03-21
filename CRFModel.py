# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:45:08 2016

@author: rsk
"""

import cPickle
import gzip
import os
import sys
import time
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import itertools
os.chdir('/home/rsk/Documents/PyStruct/CRF-MNIST-ImageDenoising/')
from pystruct.models import GraphCRF, LatentNodeCRF
from pystruct.learners import NSlackSSVM, OneSlackSSVM, LatentSSVM
from pystruct.datasets import make_simple_2x2
from pystruct.utils import make_grid_edges, plot_grid


from CRFUtils import *

def viewImg(img):
    plt.imshow( np.reshape(img,(28,28)) ,cmap=cm.Greys )
    
    
def edges(shape=(28,28),dist=4,diag=0):
    
    length = shape[0]*shape[1]
    mat = np.reshape( list(range(length)), shape  )
    mat = np.array(mat)  
    
    edgeList=[]
    
    if diag==1:
        for i in range(shape[0]):
            for j in range(shape[1]):
    
                for x in range(i,i+dist+1):
                    for y in range(j-dist,j+dist+1):
    
                        if x==i and y==j:    #Avoid edge to self
                            continue
                        if x<shape[0] and y>=0 and y<shape[1]:  #Avoid going out of the matri
                            edgeList.append(np.sort([mat[i,j] , mat[x,y] ]))
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                
                x=i
                for y in range(j-dist,j+dist+1):
                    if x<shape[0] and y>=0 and y<shape[1] and y!=j:  #Avoid going out of the matri
                        edgeList.append(np.sort([mat[i,j] , mat[x,y] ]))
                    
                for x in range(i+1,i+dist+1):
                    y=j
                    if x<shape[0] and y>=0 and y<shape[1]:  #Avoid going out of the matri
                        edgeList.append(np.sort([mat[i,j] , mat[x,y] ]))
        
    
    edgeList =np.array( sorted( np.vstack({tuple(row) for row in edgeList}), key=lambda x : x[0] )     )
    
    return np.array(edgeList)

#%%
train,valid,test = getDigitData(0)

trainDirty = addNoise(train,0.05)
testDirty = addNoise(test,0.05)

# Creating truth labels by thresholding
threshold= 0.3
trainLabels = []
testLabels = []

for i in train:
    trainLabels.append([ 1 if k>threshold else 0 for k in i ])

for i in test:
    testLabels.append([ 1 if k>threshold else 0 for k in i ])
    
trainLabels = np.array(trainLabels)
testLabels = np.array(testLabels)

#%%

edgeList = edges((28,28),dist=1,diag=0)

n_train=10
n_test=10

G = [edgeList for x in trainDirty[0:n_train]]

X_flat = [np.vstack(i) for i in trainDirty[0:n_train]]
X_flat = np.array(trainLabels[0:n_train])

crf = GraphCRF(inference_method="ad3")
svm = NSlackSSVM(model=crf,max_iter=10,C=0.1,n_jobs=-1,verbose=1)

asdf = zip(X_flat,G)
svm.fit(asdf,Y_flat)

#%%

G2 = [all_edges for x in testDirty[0:n_test]]

X_flat2 = [np.vstack(i) for i in testDirty[0:n_test]]
Y_flat2 = np.array(testLabels[0:n_test])

asdf2 = zip(X_flat2,G2)

predTrain = svm.predict(asdf)
errTrain = accuracy(predTrain,Y_flat)

predTest = svm.predict(asdf2)
errTest = accuracy(predTest,Y_flat2)

print "The train set DICE is %f" %(errTrain)
print "The test set DICE is %f" %(errTest)