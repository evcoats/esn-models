# models to compare on mixed selectivity task:
    # mixed selectivity paper: https://www.nature.com/articles/nature12160#Sec9

    # the goal is to compare a reservoir computing model to a linear model on the original mixed selectivity task

        # 1. mixed selectivity visual task model (includes non-linear reservoir): 
            # similar to section 3 of https://ieeexplore.ieee.org/document/7311148

        # 2. linear model: 


# we are using the fashion-mnist dataset for the pictures as it can essentially be any set of images to distinguish 

# we will load the first two images, then give it five sets of two images each, and it will have to do a binary classification in a size 5 array on which set comtains the sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential, save_model, load_model
from random import randrange


dataset = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = dataset.load_data()

training_images  = training_images / 255.0
test_images = test_images / 255.0


# the first two elements of X are the two images in the sequence, then the next 10 elements are a random combination of 8 other images, with the original set mixed in 

# X[0:2] == images to remember 
# X[2:11] == random images with the two images to "recognize" interspersed either once or twice
# Y[0:4] == zeros for pairs which don't include the image to recognize, ones for pairs where it does

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


number_of_pairs = 5

trainX, trainY = Sequential_Input(training_images, number_of_pairs)

testX, testY = Sequential_Input(test_images, number_of_pairs)



print(trainX[0].shape)
print(trainY[0].shape)


model1 = tf.keras.models.Sequential([tf.keras.layers.InputLayer((2*(number_of_pairs+1), 28, 28)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(20, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])


model1.compile(optimizer = tf.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model1.summary()

model1.fit(trainX, trainY, epochs=10)

losses_df1 = pd.DataFrame(model1.history.history)

losses_df1.plot(figsize = (10,6))

test_predictions1 = model1.predict(testX)

print("Test ANN Model:")
print(testY[0])

print("Prediction ANN Model")
print(test_predictions1[0])

print("Test ANN Model:")
print(testY[1])

print("Prediction ANN Model:")
print(test_predictions1[1])

trainX = trainX.reshape(trainX.shape[0],2*(number_of_pairs+1), 28*28)
testX = testX.reshape(testX.shape[0],2*(number_of_pairs+1), 28*28)

model2 = Sequential([tf.keras.layers.InputLayer(input_shape=(2*(number_of_pairs+1), 28*28)),
                                    tfa.layers.ESN(units=500,spectral_radius=0.8,input_shape=(2*(number_of_pairs+1), 28*28)),
                                    tf.keras.layers.Dense(20, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])


model2.compile(optimizer = tf.optimizers.Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model2.summary()

model2.fit(trainX, trainY, epochs=10)

losses_df1 = pd.DataFrame(model2.history.history)

losses_df1.plot(figsize = (10,6))

test_predictions2 = model2.predict(testX)

save_model(model1, "ESN_Models/MSlinear.h5")

save_model(model2, "ESN_Models/MSRC.h5")

print("Test RC Model:")
print(testY[0])

print("Prediction RC Model")
print(test_predictions2[0])

print("Test RC Model:")
print(testY[1])

print("Prediction RC Model:")
print(test_predictions2[1])

print()

numHit = 0 
for i in range(test_predictions1.shape[0]):
    indexOfPair = 0
    for j in range(number_of_pairs):
        if testY[i,j] == 1:
            indexOfPair = j
    if (max(test_predictions1[i]) == test_predictions1[i,indexOfPair]):
        numHit+=1

accuracy1 = numHit / test_predictions1.shape[0]

numHit = 0 
for i in range(test_predictions2.shape[0]):
    indexOfPair = 0
    for j in range(number_of_pairs):
        if testY[i,j] == 1:
            indexOfPair = j
    if (max(test_predictions2[i]) == test_predictions2[i,indexOfPair]):
        numHit+=1

accuracy2 = numHit / test_predictions2.shape[0]

print("accuracy of ANN:")
print(accuracy1)

print()

print("accuracy of RC:")
print(accuracy2)

