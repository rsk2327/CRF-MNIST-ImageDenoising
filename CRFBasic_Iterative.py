# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 02:01:00 2016

@author: roshan

In this model, we iteratively save the model in between the training process. This allows us to access the model weights at different
iterations of model allowing us to choose the model that overfits the least. This also allows us to track the performance of the 
model on the test dataset across iterations
"""


import cPickle
import gzip
import os
import sys
import time
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cPickle as pickle
import re
import shutil

import itertools
os.chdir('/home/rsk/Documents/PyStruct/CRF-MNIST-ImageDenoising/')
from pystruct.models import GraphCRF, LatentNodeCRF
from pystruct.learners import NSlackSSVM, OneSlackSSVM, LatentSSVM
from pystruct.datasets import make_simple_2x2
from pystruct.utils import make_grid_edges, plot_grid,SaveLogger

from CRFUtils import *
#%%##### DATA PREPARATION ################
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

#%%######## TRAINING  #####################
"""
save_every defines the frequency at which models will be saved. All iterations that are a multiple of
save_every are saved. In addition to this, the final iteration of the model is always saved irrespective 
of whether its a multiple of save_every

The models are saved in the folder defined by folderName
"""

n_train=200
n_test=100
num_iter=8
C=0.1
dist=1
diag=0
inference="ad3"
save_every=1

folderName = os.getcwd()+"/models/"

if len(os.listdir(folderName))!=0:
    shutil.rmtree(folderName)
    os.mkdir(folderName)

if dist==0:
    G=[np.empty((0, 2), dtype=np.int) for x in trainDirty[0:n_train]]
else:
    edgeList = edges((28,28),dist=dist,diag=diag)
    G = [edgeList for x in trainDirty[0:n_train]]

X_flat = [np.vstack(i) for i in trainDirty[0:n_train]]
Y_flat = np.array(trainLabels[0:n_train])

crf = GraphCRF(inference_method=inference)
svm = NSlackSSVM(model=crf,max_iter=num_iter,C=C,n_jobs=6,verbose=1,logger=SaveLogger(folderName+"model_%d.pickle",save_every=save_every,verbose=0))

asdf = zip(X_flat,G)


svm.fit(asdf,Y_flat)

#%%####### TESTING ##################


if dist==0:
    G2 = G=[np.empty((0, 2), dtype=np.int) for x in testDirty[0:n_test]]
else:
    G2 = [edgeList for x in testDirty[0:n_test]]

X_flat2 = [np.vstack(i) for i in testDirty[0:n_test]]
Y_flat2 = np.array(testLabels[0:n_test])

asdf2 = zip(X_flat2,G2)

#%%
folders = os.listdir(folderName)
num_models = len(folders)
print "Number of models :",num_models

modelTestErrors=[]

for i in range(num_models):
    errTest=0
    iteration = int(re.findall(r'\d+',folders[i])[0])
    with open(folderName+folders[i], "rb") as f:
        svm = pickle.load(f)
    predTest = svm.predict(asdf2)
    
    for j in range(len(predTest)):
        errTest += accuracy(predTest[j],Y_flat2[j])
    errTest = errTest/float(len(predTest))
    
    modelTestErrors.append((iteration,errTest))

modelTestErrors.sort(key=lambda x:x[0])

for i in range(len(modelTestErrors)):
    print "Iteration : %d , Dice Score : %f"%(modelTestErrors[i][0],modelTestErrors[i][1])

#%%

plt.plot([x[0] for x in modelTestErrors], [x[1] for x in modelTestErrors])

