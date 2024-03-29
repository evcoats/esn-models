import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python import keras

from keras.models import Sequential, save_model, load_model
from random import randrange




dataset = tf.keras.datasets.fashion_mnist

def Sequential_Input(df_np, number_of_pairs):
    X = []
    y = []
    
    for i in range(len(df_np) - 2*(number_of_pairs+1)-2):
        row = []
        for a in df_np[i:i + 2]:
            row.append(np.asarray(a ))
        rowOfMatch = randrange(5)
        newY = []
        for j in range(number_of_pairs):
            if (rowOfMatch == j):
                for a in df_np[i:i + 2]:
                    row.append(np.asarray(a ))
                newY.append(1)
            else:
                for b in df_np[i+2*(1+number_of_pairs):i+2*(1+number_of_pairs)+2]:
                    row.append(np.asarray(b))
                newY.append(0)

        X.append(row)
        y.append(newY)



    return np.array(X), np.array(y)


(training_images, training_labels), (test_images, test_labels) = dataset.load_data()

training_images  = training_images / 255.0
test_images = test_images / 255.0

number_of_pairs = 5

trainX, trainY = Sequential_Input(training_images, number_of_pairs)

testX, testY = Sequential_Input(test_images, number_of_pairs)

# set trainX and trainY to be of size 100

trainX = trainX[:3000]
trainY = trainY[:3000]

print("training examples: 3000")

def createClassicalModel(sizeOfL2, lr):
    model1 = tf.keras.models.Sequential([tf.keras.layers.InputLayer((2*(number_of_pairs+1), 28, 28)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(sizeOfL2, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])

    print("learning rate of RC: "+ str(lr))

    model1.compile(optimizer = tf.optimizers.Adam(lr=0.1),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

    model1.summary()
    
    return model1


def createRCModel(sizeOfL2, sizeOfReservoir, lr):
        
    initializer = tf.keras.initializers.Ones()


    model2 = Sequential([tf.keras.layers.InputLayer(input_shape=(2*(number_of_pairs+1), 28*28)),
                                    # tfa.layers.ESN(units=sizeOfReservoir,spectral_radius=0.8,input_shape=(2*(number_of_pairs+1), 28*28)),
                                    tfa.layers.ESN(units=sizeOfReservoir,spectral_radius=1),

                                    tf.keras.layers.Dense(sizeOfL2, activation=tf.nn.relu),

                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])
    
    print("learning rate of RC: " + str(lr))

    model2.compile(optimizer = tf.optimizers.Adam(lr=lr),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

    model2.summary()

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

print("size of L2: "+ str(50))

classicalModel = createClassicalModel(50, 0.001)
classicalModel.fit(trainX, trainY, epochs=10)
classicAccuracy = modelAccuracy(classicalModel.predict(testX), testY)

trainX2 = trainX.reshape(trainX.shape[0],2*(number_of_pairs+1), 28*28)
testX2 = testX.reshape(testX.shape[0],2*(number_of_pairs+1), 28*28)
    
print("Classic Accuracy: "+ str(classicAccuracy))
print(number_of_pairs)
print(trainX2.shape[0])

RCModel = createRCModel(50,28*28,0.001)

print(trainX2.shape)

RCModel.fit(trainX2, trainY, epochs=10)

RCAccuracy = modelAccuracy(RCModel.predict(testX2), testY)

print("RC accuracy: "+ str(RCAccuracy))

    




