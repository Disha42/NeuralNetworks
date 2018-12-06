# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 19:49:42 2018

@author: disna
"""

import numpy as np
import math
import sys



with open(sys.argv[1], "r") as ins:
    count =0
        
    trainFeatures =[]    
    for line in ins:
        if count!=0:
            values = line.split(",")
            features=[]
            for val in values:
                feature =val.rstrip()
                if feature=='yes':
                    feature =1
                if feature =="no":
                    feature =0
                feature = float(feature)
                features.append(feature)
            trainFeatures.append(np.array(features))
        count +=1
    trainFeatures = np.vstack(trainFeatures)
    
    trainFeatures[:,0] = (trainFeatures[:,0]-1900)/100
    trainFeatures[:,1] = (trainFeatures[:,1])/7
    x_normed = trainFeatures
    
    a = np.ones((len(trainFeatures),1 ))
    x_normed = np.hstack((a,x_normed))
    
    
with open(sys.argv[2], "r") as label_file:
        train_labels =[]
        for line in label_file:
            if line.rstrip() =='yes':
                train_labels.append(1)
            elif line.rstrip() =='no':
                train_labels.append(0)
        
        train_labels = np.array(train_labels) 
        train_labels = train_labels.reshape(100,1)
        
with open(sys.argv[4], "r") as weights0:
    weightsVector0 =[]
    
    for line in weights0:
        values = line.split(",")
        weights=[]
        for val in values:
            weights.append(float(val.rstrip()))
        weightsVector0.append(np.array(weights))
    weightsVector0 = np.vstack(weightsVector0)
    weightsVector0 = weightsVector0.T
    
    weightsVector0Grad = weightsVector0
    #print("weightsVector0")
    #print(weightsVector0)
   
    
    
with open(sys.argv[5], "r") as weights1:
    weightsVector1 =[]
    
    for line in weights1:
        values = line.split(",")
        weights=[]
        for val in values:
            weights.append(float(val.rstrip()))
        weightsVector1.append(np.array(weights))
    weightsVector1 = np.vstack(weightsVector1)
    weightsVector1Grad = weightsVector1
    #print("weightsVector1")
    #print(weightsVector1)
    


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)



def feedForwardGrad(weightsVector0Grad,weightsVector1Grad):

    errors = 0
    hiddenOutputGrad =np.zeros((100,3))
    finalOutputGrad =np.zeros(100)
    oneColumn = np.ones(100).reshape(100,1)
    
    #print("shape of weightsVector0Grad" +str(weightsVector0Grad.shape))
    
    #print(x_normed)
    hiddenOutputGrad = sigmoid_v(np.dot(x_normed,weightsVector0Grad.T))
            
    hiddenOutputGrad = np.hstack([oneColumn, hiddenOutputGrad])
    #print("shape of hiddenOutputGrad" +str(hiddenOutputGrad.shape))
    
    #print(hiddenOutputGrad)

    
                
    finalOutputGrad = sigmoid_v(np.dot(weightsVector1Grad.T,hiddenOutputGrad.T))
    
    #print(finalOutputGrad)
    #print("************")
    finalOutputGrad = finalOutputGrad.T
    
    #print(finalOutputGrad)
     
    #print(train_labels)      
    error = np.power((train_labels-finalOutputGrad),2)  
    #print(error)      
    errors = np.sum(error)

    errors = errors/2
       
    #print("hiddenOutputGrad is...")
    #print(hiddenOutputGrad)
            
    (weightsVector0Grad_New,weightsVector1Grad_New)= backPropGrad(weightsVector0Grad,weightsVector1Grad,train_labels,finalOutputGrad,x_normed,hiddenOutputGrad)
    
    
    return (weightsVector0Grad_New,weightsVector1Grad_New,errors)

def backPropGrad(weightsVector0Grad,weightsVector1Grad, target, predicted,x,hiddenOutputGrad):
    
    #print("in back prop")
    


    del0 = predicted*(1-predicted)*(target-predicted)
    
    #print(del0)
    
    delWeight0 = 0.1 * np.dot(del0.T,hiddenOutputGrad)
    
    OneMinus = 1-hiddenOutputGrad
    
    part1 = (hiddenOutputGrad * OneMinus )
    
    del1 = part1 * np.dot(del0,weightsVector1Grad.T)
    
    weightsVector1Grad =  weightsVector1Grad + (delWeight0.T)
    
    del1 = np.delete(del1, 0, 1)
    
    delWeight1 = 0.1 * (np.dot(del1.T, x))
    
    weightsVector0Grad = weightsVector0Grad + delWeight1
    
    return (weightsVector0Grad,weightsVector1Grad)





def feedForwardStochastic(weightsVector0,weightsVector1):
    count =0
    errors = 0
    hiddenOutput =[0,0,0,0]
    finalOutput =0
    for x in x_normed:

        x = x.reshape(1,x.shape[0])
        hiddenOutput = sigmoid_v(np.dot(x,weightsVector0.T))
            
        hiddenOutput = hiddenOutput.reshape((3,1))
        hiddenOutput = np.vstack([[1], hiddenOutput])
            
        #if count ==0:
                #print("hiddenOutput is..")
                #print(hiddenOutput)
               # print("weightsVector1 is..")
               # print(weightsVector1.T)
                
        finalOutput = sigmoid(np.dot(weightsVector1.T,hiddenOutput))
            
                
            
        error = ((train_labels[count]-finalOutput)**2)

            
        errors += error
        
            
        (weightsVector0,weightsVector1)= backPropStochastic(weightsVector0,weightsVector1,train_labels[count],finalOutput,x,hiddenOutput)
        count+=1 
    errors = errors/2
    
    
    return (weightsVector0,weightsVector1,errors[0])

            

    
    
    
def backPropStochastic(weightsVector0,weightsVector1, target, predicted,x,hiddenOutput):
    
    #print("in back prop")
    del0 = predicted*(1-predicted)*(target-predicted)
    
    #print(del0)
    
    #print(weightsVector1.shape)
    #print(hiddenOutput)
    
    #print(weightsVector1)
    delWeight0 = 0.4 * del0* (hiddenOutput.reshape(weightsVector1.shape))
    
    part1 = (hiddenOutput * [1-hiddenOutput] ).reshape(weightsVector1.shape)
    
    #print(part1)
    
    del1 = part1 * weightsVector1 *del0
    
    #print("delWeight1 is ")
    
    #print(delWeight0)
    
    weightsVector1 =  weightsVector1 + delWeight0
    
    #print(weightsVector1)
    
    
    #print("del1 is..")
    #print(del1)
    
    #print(del1.shape)
    
    del1 = np.delete(del1, 0, 0)
    
    #print("del1 is after deleting 1 st row..")
    #print(del1)
    
    #print(del1.shape)
    
    delWeight1 = 0.4 * (np.dot(del1, x))
    #print(delWeight1)
    
    weightsVector0 = weightsVector0 + delWeight1
    
    #print(weightsVector0)
    
    return (weightsVector0,weightsVector1)



    

def predict(finalweightsVector0,finalweightsVector1,xdev_normed):
    hiddenOutput =[0,0,0,0]
    finalOutput =0
    for x in xdev_normed:

        x = x.reshape(1,x.shape[0])
        hiddenOutput = sigmoid_v(np.dot(x,finalweightsVector0.T))
            
        hiddenOutput = hiddenOutput.reshape((3,1))
        hiddenOutput = np.vstack([[1], hiddenOutput])
                
        finalOutput = sigmoid(np.dot(finalweightsVector1.T,hiddenOutput))
        if finalOutput<0.5:
            print("no")
        else:
            print("yes")

            
                
            
        
    
    

    
    
prev= 10000000  
for i in range(1,1000):        
    (weightsVector0Grad,weightsVector1Grad,errors) = feedForwardGrad(weightsVector0Grad,weightsVector1Grad)
    if errors< prev:
        print(errors)
        prev = errors
    else:
        break
#finalweightsVector0,finalweightsVector1 = weightsVector0Grad,weightsVector1Grad   
    
    
print("GRADIENT DESCENT TRAINING COMPLETED!")
        
for i in range(1,2000):        
    (weightsVector0,weightsVector1,errors) = feedForwardStochastic(weightsVector0,weightsVector1)
    if i <16:
        print(errors)
finalweightsVector0,finalweightsVector1 = weightsVector0,weightsVector1


    
print("STOCHASTIC GRADIENT DESCENT TRAINING COMPLETED! NOW PREDICTING.")

with open(sys.argv[3], "r") as ins:
    count =0
        
    devFeatures =[]    
    for line in ins:
        if count!=0:
            values = line.split(",")
            features=[]
            for val in values:
                feature =val.rstrip()
                if feature=='yes':
                    feature =1
                if feature =="no":
                    feature =0
                feature = float(feature)
                features.append(feature)
            devFeatures.append(np.array(features))
        count +=1
    devFeatures = np.vstack(devFeatures)
    
    devFeatures[:,0] = (devFeatures[:,0]-1900)/100
    devFeatures[:,1] = (devFeatures[:,1])/7
    xdev_normed = devFeatures
    
    a = np.ones((len(devFeatures),1 ))
    xdev_normed = np.hstack((a,xdev_normed))
    predict(finalweightsVector0,finalweightsVector1,xdev_normed)

   



    
        
        
        
        
        
        
    
    

            
         
    



    