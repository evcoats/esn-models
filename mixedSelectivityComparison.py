import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential, save_model, load_model
from random import randrange
from mixedSelectivityTask import Sequential_Input

dataset = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = dataset.load_data()

training_images  = training_images / 255.0
test_images = test_images / 255.0

number_of_pairs = 5

trainX, trainY = Sequential_Input(training_images, number_of_pairs)

testX, testY = Sequential_Input(test_images, number_of_pairs)

def createClassicalModel(sizeOfL2):
    model1 = tf.keras.models.Sequential([tf.keras.layers.InputLayer((2*(number_of_pairs+1), 28, 28)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(sizeOfL2, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])


    model1.compile(optimizer = tf.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

    # model1.summary()
    
    return model1


def createRCModel(sizeOfL2, sizeOfReservoir):
    model2 = Sequential([tf.keras.layers.InputLayer(input_shape=(2*(number_of_pairs+1), 28*28)),
                                    # tfa.layers.ESN(units=sizeOfReservoir,spectral_radius=0.8,input_shape=(2*(number_of_pairs+1), 28*28)),
                                    tfa.layers.ESN(units=sizeOfReservoir,spectral_radius=0.8),
                                    tf.keras.layers.Dense(sizeOfL2, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])
    
    model2.compile(optimizer = tf.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

    # model2.summary()

    return model2

def modelAccuracy(predictions, testY):
    numHit = 0 
    for i in range(predictions.shape[0]):
        indexOfPair = 0
        for j in range(number_of_pairs):
            if testY[i,j] == 1:
                indexOfPair = j
        if (max(predictions[i]) == predictions[i,indexOfPair]):
            numHit+=1

    return numHit / predictions.shape[0]

accuracyArray = []

for sizeOfL2Iter in range(5):
    print("size of L2: "+ str(sizeOfL2Iter*10+10))
    classicalModel = createClassicalModel(sizeOfL2Iter*10+10)
    classicalModel.fit(trainX, trainY, epochs=10)
    classicAccuracy = modelAccuracy(classicalModel.predict(testX), testY)
    
    trainX2 = trainX.reshape(trainX.shape[0],2*(number_of_pairs+1), 28*28)
    testX2 = testX.reshape(testX.shape[0],2*(number_of_pairs+1), 28*28)

    print("Classic Accuracy: "+ str(classicAccuracy))

    row = []
    row.append(classicAccuracy)

    for sizeOfReservoirIter in range(5):
        print("size of reservoir: "+str(sizeOfReservoirIter*100+100) )
        print(number_of_pairs)
        print(trainX2.shape[0])
        
        RCModel = createRCModel(sizeOfL2Iter*10+10,sizeOfReservoirIter*100+100)
        
        print(trainX2.shape)

        RCModel.fit(trainX2, trainY, epochs=10)

        RCAccuracy = modelAccuracy(RCModel.predict(testX2), testY)

        print("RC accuracy: "+ str(RCAccuracy))
        row.append(RCAccuracy)

    accuracyArray.append(row)

print(accuracyArray)

    




