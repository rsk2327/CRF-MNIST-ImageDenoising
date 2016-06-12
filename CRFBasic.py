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


#%%
"""
Use getDigitData to get images of a particular digit from the entire MNIST dataset

To increase the difficulty of the denoising task, increase the noise parameter of addNoise

"""
train,valid,test = getDigitData(0)

trainDirty = addNoise(train,noise=0.05)
testDirty = addNoise(test,noise=0.05)

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
"""
To train a model with only unary potentials, set dist = 0
"""


n_train=200
n_test=100
num_iter=10
C=0.1
dist=0
diag=0
inference="ad3"



if dist==0:
    G=[np.empty((0, 2), dtype=np.int) for x in trainDirty[0:n_train]]
else:
    edgeList = edges((28,28),dist=dist,diag=diag)
    G = [edgeList for x in trainDirty[0:n_train]]

X_flat = [np.vstack(i) for i in trainDirty[0:n_train]]
Y_flat = np.array(trainLabels[0:n_train])

crf = GraphCRF(inference_method=inference)
svm = NSlackSSVM(model=crf,max_iter=num_iter,C=C,n_jobs=6,verbose=1)

asdf = zip(X_flat,G)

svm.fit(asdf,Y_flat)

#%%

if dist==0:
    G2 = G=[np.empty((0, 2), dtype=np.int) for x in testDirty[0:n_test]]
else:
    G2 = [edgeList for x in testDirty[0:n_test]]

X_flat2 = [np.vstack(i) for i in testDirty[0:n_test]]
Y_flat2 = np.array(testLabels[0:n_test])

asdf2 = zip(X_flat2,G2)

predTrain = svm.predict(asdf)
errTrain = 0
for i in range(len(predTrain)):
    errTrain += accuracy(predTrain[i],Y_flat[i])
errTrain = errTrain/float(len(predTrain))

predTest = svm.predict(asdf2)
errTest = 0
for i in range(len(predTest)):
    errTest += accuracy(predTest[i],Y_flat2[i])
errTest = errTest/float(len(predTest))


print "The train set DICE is %f" %(errTrain)
print "The test set DICE is %f" %(errTest)

#%%


resultsDir = os.getcwd()+"/Results"
resultFile  = open(resultsDir + "/results.csv",'a')
resultFile.write(str(num_iter)+","+str(dist)+","+str(diag)+","+inference+","+str(errTrain)+","+str(errTest)+"\n")
resultFile.close()

nameLen = len(os.listdir(resultsDir))
filename = str(nameLen)+"_"+str(dist)+"_"+str(diag)+"_"+inference
predFileTrain = open(resultsDir+"/"+filename+"_Train.pkl",'wb')
predFileTest = open(resultsDir+"/"+filename+"_Test.pkl",'wb')
cPickle.dump(predTrain,predFileTrain,protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(predTest,predFileTest,protocol=cPickle.HIGHEST_PROTOCOL)
predFileTrain.close()
predFileTest.close()
