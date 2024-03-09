# models to compare on mixed selectivity task:
    # mixed selectivity paper: https://www.nature.com/articles/nature12160#Sec9

    # the goal is to compare a reservoir computing model to a linear model on the original mixed selectivity task

        # 1. mixed selectivity visual task model (includes non-linear reservoir): 
            # similar to section 3 of https://ieeexplore.ieee.org/document/7311148

            # - input layer
            # - 3 reservoir layers: 
                # layer 1: (λ,αU,ρ,Kin,Krec)=(0.22,0.28,0.65,5,5)
                # next reservoir layers: (λ,αU,ρ,Kin,Krec)=(0.22, 0.6, 0.4, 5, 5)
            # - 3 readout layers (after each reservoir layer):
                # 50 readout nodes each
            # - one classification layer
                # 5 nodes each

        # 2. linear model: 
            # - input layer
            # - 3 fully connected layers, 50 each, 
                # 50 readout nodes
            # backpropagation to train


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

def Sequential_Input(df, number_of_pairs):
    df_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_np) - 2*(number_of_pairs+1)):
        row = []
        row1 = [a for a in df_np[i:i + 2]]
        row.append(row1)
        rowOfMatch = randrange(5)
        newY = []
        for j in range(number_of_pairs):
            if (rowOfMatch == j):
                row.append(row1)
                newY.append(1)
            else:
                row.append([a for a in df_np[i+2(1+number_of_pairs):i+2(1+number_of_pairs)+2]])
                newY.append(0)

        X.append(row)
        y.append(newY)

    return np.array(X), np.array(y)

number_of_pairs = 5

trainX, trainY = Sequential_Input(training_images, number_of_pairs)

testX, testY = Sequential_Input(test_images, number_of_pairs)

model = tf.keras.models.Sequential([tf.keras.layers.InputLayer((2*(number_of_pairs+1), 28, 28)),
                                    tf.keras.layers.Flatten(),tf.keras.layers.Dense(100, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(100, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(100, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])


model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)

losses_df1 = pd.DataFrame(model.history.history)

losses_df1.plot(figsize = (10,6))

test_predictions1 = model.predict(testX).flatten()

print("Test:")
print(testY[0])

print("Prediction")
print(test_predictions1[0])

X_test_list = []
for i in range(len(testX)):
    X_test_list.append(testX[i][0])
    
test_predictions_df1 = pd.DataFrame({'X_test':list(X_test_list), 
                                    'ESN Prediction':list(test_predictions1)})

test_predictions_df1.plot(figsize = (15,6))

plt.show()

test_predictions_df1[(len(testX) - 720):].plot(figsize = (15,5))
test_predictions_df1.plot(figsize = (15,6))

plt.show()








