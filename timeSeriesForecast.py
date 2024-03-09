import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential, save_model, load_model

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


df = pd.read_csv(csv_path)
df_hour_lvl = df[5::6].reset_index().drop('index', axis=1)

print(df_hour_lvl.head())

def Sequential_Input(df, input_sequence):
    df_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)
        
    return np.array(X), np.array(y)


n_input = 10      

df_min_model_data = df_hour_lvl['T (degC)']

X, y = Sequential_Input(df_min_model_data, n_input)

# Training data
X_train, y_train = X[:60000], y[:60000]

# Validation data
X_val, y_val = X[60000:65000], y[60000:65000]

# Test data
X_test, y_test = X[65000:], y[65000:]

print("X/Y train: ")
      
print(X_train[0])

print(y_train[0])

n_features = 1

model = Sequential([
    tf.keras.layers.InputLayer((n_input,n_features)),
    tfa.layers.ESN(units=100,spectral_radius=0.8),
    tf.keras.layers.Dense(units=1, activation = 'linear')
])

model.summary()

optimizers = [
    tf.keras.optimizers.Adam(learning_rate=0.001),
]

optimizers_and_layers = [(optimizers[0], model.layers[1])]

optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

model.compile(loss = 'mean_absolute_error',optimizer=optimizer)

model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 50)

losses_df1 = pd.DataFrame(model.history.history)

losses_df1.plot(figsize = (10,6))

save_model(model, "ESN_Models/BasicForecast.h5")

model = load_model('ESN_Models/BasicForecast.h5')

test_predictions1 = model.predict(X_test).flatten()

print("Test:")
print(X_test[0])

print("Prediction:")
print(test_predictions1[0])

X_test_list = []
for i in range(len(X_test)):
    X_test_list.append(X_test[i][0])
    
test_predictions_df1 = pd.DataFrame({'X_test':list(X_test_list), 
                                    'ESN Prediction':list(test_predictions1)})

test_predictions_df1.plot(figsize = (15,6))

plt.show()

test_predictions_df1[(len(X_test) - 720):].plot(figsize = (15,5))
test_predictions_df1.plot(figsize = (15,6))

plt.show()
