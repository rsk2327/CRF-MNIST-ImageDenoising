# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:43:20 2016

@author: rsk
"""


import cPickle
import gzip
import os
import sys
import time
import numpy as np
import itertools
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib

def plotImg(data):
    plt.imshow(data,cmap=cm.Greys)
    
def reshape(data):
    return np.reshape(data,(28,28))

def sp_noise(image,prob):
    output = np.zeros(image.shape)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
            else:
                output[i][j] = image[i][j]
    return output

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
    



def gaussianNoise(shape):
    return np.random.normal(loc=0.5,scale=1,size=shape)
    

def error(y, truth):
    
    corr=0.0
    total=0.0
    
    for i in range(len(y)):
        corr += np.sum(y[i]==truth[i])
        
        total += len(y[i])
       
        
    return (corr/total)*100


def confusion(y,truth):
    
    
    a = zip(y,truth)
    tp = np.sum(np.array([int(i==(1,1)) for i in a]))
    
    tn = np.sum(np.array([int(i==(0,0)) for i in a]))
    fp = np.sum(np.array([int(i==(1,0)) for i in a]))
    fn = np.sum(np.array([int(i==(0,1)) for i in a]))
    
    print "tp : "+str(tp)+"  fp : "+str(fp)
    print "tp : "+str(fn)+"  tn : "+str(tn)
   
    return float((tp+tn))/float(len(y))*100.0

def addEdges(shape,diag=0,dist=1):
    edges=[]
    #diag_edges=[]
    
    for distance in range(1,dist+1):
        ##Adding horizontal edges    
        for i in range( shape[0]):
            for j in range(shape[1]-distance):
                edges.append([i*shape[0] + j, i*shape[0] +j +distance ])
                
        ## Adding vertical edges
        for i in range( shape[0]-distance):
            for j in range(shape[1]):
                edges.append([i*shape[0] + j, (i+distance)*shape[0] + j])
                
    if diag==1:
        for distance in range(1,dist+1):
        
            #Adding up-right edges:
            for i in range(shape[0]-distance):
                for j in range(shape[1] - distance):
                    edges.append([i*shape[0]+j , (i+distance)*shape[0] + j+distance])
                
        # Adding up-left edges:
            for i in range(shape[0]-distance):
                for j in range(distance,shape[1]):
                    edges.append([i*shape[0]+j , (i+distance)*shape[0] + j- distance])
    
    return np.array(edges)
    
    
def gen3Dedges(shape,diag=0,dist=1):
    
    
    edges=[]
    
    #adding horizontal edges
    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]-1):
                edges.append([z*shape[1]*shape[0] + y*shape[0] + x,z*shape[1]*shape[0] + y*shape[0] + x + 1])
                
    #adding vertical edges
    for z in range(shape[2]):
        for y in range(shape[1]-1):
            for x in range(shape[0]):
                edges.append([z*shape[1]*shape[0] + y*shape[0] + x,z*shape[1]*shape[0] + (y+1)*shape[0] + x ])
    
    #adding top edges                
    for z in range(shape[2]-1):
        for y in range(shape[1]):
            for x in range(shape[0]):
                edges.append([z*shape[1]*shape[0] + y*shape[0] + x,(z+1)*shape[1]*shape[0] + y*shape[0] + x ])
    
    
    return edges
    
    
        
#Function to add noise to the entire dataset
def addNoise(x,noise):
    dtrain=[]
    
    for i in x:
        dtrain.append(np.hstack(sp_noise(reshape(i), noise)))
        
    return dtrain
    
def getDigitData(digit=0):
    path = os.getcwd() + "/mnist.pkl.gz"

    f = gzip.open(path,'rb')
    train,valid,test = cPickle.load(f)
    f.close()
    
    trainx,trainy = train
    validx,validy = valid
    testx,testy= test
    
    train= []
    valid=[]
    test=[]
    
    for i in range(len(trainx)):
        if trainy[i]==digit:
            train.append(trainx[i])
    for i in range(len(validx)):
        if validy[i]==digit:
            valid.append(validx[i])
        
    for i in range(len(testx)):
        if testy[i]==digit:
            test.append(testx[i])
            
    return (train,valid,test)


def checkUp(x,y,shape=(28,28)):
    if x>=0 and x<shape[0] and y>=0 and y<shape[1]:
        return 1
    else:
        return 0

def getFeatures(img,shape,dist=1):
    features=[]
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            
            
            data=[]
            data.append(img[i*shape[0]+j])
            #adding right
            for distance in range(1,dist+1):
                if checkUp(i,j+distance,shape):
                    #print('y1')
                    data.append(img[i*shape[0]+j+distance])
                else:
                    data.append(0)
                
            #adding left
            for distance in range(1,dist+1):
                if checkUp(i,j-distance,shape):
                    #print('y2')
                    data.append(img[i*shape[0]+j-distance])
                else:
                    data.append(0)
                    
            #adding up
            for distance in range(1,dist+1):
                if checkUp(i-distance,j,shape):
                    #print('y3')
                    data.append(img[(i-distance)*shape[0]+j])
                else:
                    data.append(0)
                    
            #adding bottom
            for distance in range(1,dist+1):
                if checkUp(i+distance,j,shape):
                    #print('y4')
                    data.append(img[(i+distance)*shape[0]+j])
                else:
                    data.append(0)
            #print data
            features.append(data)
        
    return np.array(features)
        
        
    
    
def extractSlices(folderPath,numImages,padding=(0,0,0,0)):
    
    post=[]
    truth=[]
    flair=[]
    t1=[]
    t1c=[]
    t2=[]
    
    ############## Loading the volumes #################################
    counter=0
    for folder in os.listdir(folderPath):
        path = folderPath + "/" + folder
        print path
        
        names=os.listdir(path)
        for i in range(len(names)):
            if "new_n4dropout_" in names[i] and ".npy" in names[i]:
                data = np.load(path+"/"+names[i])[padding[0]:(240-padding[1]) , padding[2]:(240-padding[3] ),: ]
                post.append(data)
            if "binary_truth_whole_tumor" in names[i]:
                data = np.array(nib.load(path+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
                truth.append(data)
            if "Flair" in names[i] and ".nii" in names[i]:
                data = np.array(nib.load(path+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
                flair.append(data)
            if "T2" in names[i]:
                data = np.array(nib.load(path+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
                t2.append(data)
            if "T1c" in names[i]:
                data = np.array(nib.load(path+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
                t1c.append(data)
            if "T1" in names[i] and not "T1c" in names[i]:
                data = np.array(nib.load(path+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
                t1.append(data)
                
        counter+=1
        if counter==numImages and numImages!=-1:
            break
        
    
    
    ######### Extracting the slices ###############################
    postLayers=[]
    truthLayers=[]
    flairLayers=[]
    t1Layers=[]
    t1cLayers=[]
    t2Layers=[]
    
    sliceShape = truth[0][:,:,0].shape
    
    for i in range(len(post)):
        
        for j in range(155):
            a = truth[i][:,:,j].reshape((sliceShape[0]*sliceShape[1],) )
            if a.sum()>0:
                start=j+4
                break
            
        for j in range(154,-1,-1):
            a = truth[i][:,:,j].reshape((sliceShape[0]*sliceShape[1],) )
            if a.sum()>0:
                end = j-4
                break
            
        if np.mean( post[i][:,:,start][:,:,0] )<0.5:
            continue
        for j in range(start,end):
            postLayers.append( post[i][:,:,j])
            truthLayers.append(np.flipud(np.rot90(truth[i][:,:,j],1)) )
            flairLayers.append( np.flipud(np.rot90(flair[i][:,:,j],1)))
            t1Layers.append( np.flipud(np.rot90(t1[i][:,:,j],1)))
            t1cLayers.append( np.flipud(np.rot90(t1c[i][:,:,j],1)))
            t2Layers.append( np.flipud(np.rot90(t2[i][:,:,j],1)))
            
    ####### Cleaning truth and post slices ###################

    print "Making max prob predictions"            
    postLayersPred=[]                
    for i in range( len(truthLayers) ):
        postLayersPred.append(np.zeros(sliceShape))
        for j in range(sliceShape[0]):
            for k in range( sliceShape[1]):
                if postLayers[i][j,k][0]>np.max(postLayers[i][j,k][1:5]):
                    postLayersPred[i][j,k]=0.0
                else:
                    postLayersPred[i][j,k]=1.0


    print "cleaning post data and appending image data"
    for i in range( len(truthLayers) ):
        
        for j in range(sliceShape[0]):
            for k in range( sliceShape[1]):
                if np.array_equal(postLayers[i][j,k], np.array([0.0,0.0,0.0,0.0,0.0])):
                    postLayers[i][j,k]=np.array([1.0,0.0,0.0,0.0,0.0])
        postLayers[i] = np.dstack((postLayers[i],flairLayers[i],t1Layers[i],t1cLayers[i],t2Layers[i],postLayersPred[i]))
                
                
    for i in range( len(truthLayers)):
        for j in range(sliceShape[0]):
            for k in range(sliceShape[1]):
                if truthLayers[i][j,k]>0.0:
                    truthLayers[i][j,k]=1.0
    print "truthLayers length is %f" %(len(truthLayers))
    
#    Layers=[]
#    print "appending image layers"
#    for index in range(len(truthLayers)):       
#        output=[]
#        
#        for i in range(sliceShape[0]):
#            output.append([])
#            for j in range(sliceShape[1]):            
#                output[i].append([ flairLayers[index][i,j] , t1Layers[index][i,j] , t1cLayers[index][i,j] , t2Layers[index][i,j] ])
#        
#        Layers.append(output)
#    print "Layers length : %f " % (len(Layers))
                    
                 
    return [np.array(postLayers),truthLayers,sliceShape]


def extractTestSlices(folderPath,padding=(0,0,0,0)):

    
    
    names=os.listdir(folderPath)
    for i in range(len(names)):
        if "new_n4dropout_" in names[i] and ".npy" in names[i]:
            post = np.load(folderPath+"/"+names[i])[padding[0]:(240-padding[1]) , padding[2]:(240-padding[3] ),: ]
            
        if "binary_truth_whole_tumor" in names[i]:
            truth = np.array(nib.load(folderPath+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
            
        if "Flair" in names[i] and ".nii" in names[i]:
            flair = np.array(nib.load(folderPath+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
            
        if "T2" in names[i]:
            t2 = np.array(nib.load(folderPath+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
            
        if "T1c" in names[i]:
            t1c = np.array(nib.load(folderPath+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
            
        if "T1" in names[i] and not "T1c" in names[i]:
            t1 = np.array(nib.load(folderPath+"/"+ names[i]).get_data())[padding[0]:(240-padding[1]), padding[2]:(240-padding[3]),:]
            
    
    postLayers=[]
    truthLayers=[]
    flairLayers=[]
    t1Layers=[]
    t1cLayers=[]
    t2Layers=[]
    
    sliceShape = truth[:,:,0].shape
    

        
    for j in range(155):
        a = truth[:,:,j].reshape((sliceShape[0]*sliceShape[1],) )
        if a.sum()>0:
            start=j+4
            break
            
    for j in range(154,-1,-1):
        a = truth[:,:,j].reshape((sliceShape[0]*sliceShape[1],) )
        if a.sum()>0:
            end = j-4
            break
            
    if np.mean( post[:,:,start][:,:,0] )<0.5:
        return 0
    for j in range(start,end):
        postLayers.append( post[:,:,j])
        truthLayers.append(np.flipud(np.rot90(truth[:,:,j],1)) )
        flairLayers.append( np.flipud(np.rot90(flair[:,:,j],1)))
        t1Layers.append( np.flipud(np.rot90(t1[:,:,j],1)))
        t1cLayers.append( np.flipud(np.rot90(t1c[:,:,j],1)))
        t2Layers.append( np.flipud(np.rot90(t2[:,:,j],1)))
        

########### Making max prob predictions from posterior prob data #############
    postLayersPred=[]                
    for i in range( len(truthLayers) ):
        postLayersPred.append(np.zeros(sliceShape))
        for j in range(sliceShape[0]):
            for k in range( sliceShape[1]):
                if postLayers[i][j,k][0]>np.max(postLayers[i][j,k][1:5]):
                    postLayersPred[i][j,k]=0.0
                else:
                    postLayersPred[i][j,k]=1.0
                    
######## Cleaning post data and appending image data##############        
   
    
    for i in range( len(truthLayers) ):
        for j in range(sliceShape[0]):
            for k in range( sliceShape[1]):
                if np.array_equal(postLayers[i][j,k], np.array([0.0,0.0,0.0,0.0,0.0])):
                    postLayers[i][j,k]=np.array([1.0,0.0,0.0,0.0,0.0])
        postLayers[i] = np.dstack((postLayers[i],flairLayers[i],t1Layers[i],t1cLayers[i],t2Layers[i],postLayersPred[i]))
                    

                    

                    

########### Cleaning truth data #######################
    for i in range( len(truthLayers)):
        for j in range(sliceShape[0]):
            for k in range(sliceShape[1]):
                if truthLayers[i][j,k]>0.0:
                    truthLayers[i][j,k]=1.0
#    print "truthLayers length is %f" %(len(truthLayers))
    
    
    return[np.array(postLayers),truthLayers,sliceShape]
    
    
def accuracy(x,y):
    return np.sum(x[y==1])*2.0/(np.sum(x) + np.sum(y))