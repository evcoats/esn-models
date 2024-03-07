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

dataset = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = dataset.load_data()

training_images  = training_images / 255.0
test_images = test_images / 255.0



model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(50, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(50, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(50, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])


model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)






