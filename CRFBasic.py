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
    
def checkPred(index,train=0):
    
    if train==0:
        plt.subplot(121)
        plt.imshow( np.reshape(trainLabels[index],(28,28)) , cmap=cm.Greys )

        plt.subplot(122)
        plt.imshow( np.reshape(predTrain[index],(28,28)) , cmap=cm.Greys )
    else:
        plt.subplot(121)
        plt.imshow( np.reshape(testLabels[index],(28,28)) , cmap=cm.Greys )
        
        plt.subplot(122)
        plt.imshow( np.reshape(predTest[index],(28,28)) , cmap=cm.Greys )
    
    
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



n_train=200
n_test=100
num_iter=40
C=0.1
dist=2
diag=0
inference="ad3"

edgeList = edges((28,28),dist=dist,diag=diag)

G = [edgeList for x in trainDirty[0:n_train]]

X_flat = [np.vstack(i) for i in trainDirty[0:n_train]]
Y_flat = np.array(trainLabels[0:n_train])

crf = GraphCRF(inference_method=inference)
svm = NSlackSSVM(model=crf,max_iter=num_iter,C=C,n_jobs=6,verbose=1)

asdf = zip(X_flat,G)
svm.fit(asdf,Y_flat)

#%%

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
resultFile.write(str(num_iter)+","+str(dist)+","+str(diag)+","+inference+","+str(errTrain)+","+str(errTest)+","+"3x3neighbor"+"\n")
resultFile.close()

nameLen = len(os.listdir(resultsDir))
filename = str(nameLen)+"_"+str(dist)+"_"+str(diag)+"_"+inference+"_"+"3x3neighbor"
predFileTrain = open(resultsDir+"/"+filename+"_Train.pkl",'wb')
predFileTest = open(resultsDir+"/"+filename+"_Test.pkl",'wb')
cPickle.dump(predTrain,predFileTrain,protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(predTest,predFileTest,protocol=cPickle.HIGHEST_PROTOCOL)
predFileTrain.close()
predFileTest.close()
