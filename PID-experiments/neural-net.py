import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras import backend as K
import os 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import datetime
from datetime import datetime
import sys

# LOAD DATA
data = pd.read_csv('..\..\pidd-preprocessed.csv', sep='\s*,\s*', header=0, names=['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age', 'Outcome'])

# SET RANDOM SEED
# Choose seed value (Max accuracy seed: 1657221665)
dt_int = int(datetime.utcnow().timestamp())


# Generate random batch size
"""import random
batch_seed = int(datetime.utcnow().timestamp())
random.seed(batch_seed)
nn_batch_size = random.randint(1, len(data))"""

# Set batch size (Max accuracy batch size: 13)
# From command line
"""i, arg = sys.argv
nn_batch_size = int(arg)"""

nn_batch_size = 1
# Save batch size to file
batch_size_file = open(r"batch_size_file.txt", "w")
batch_size_file.write(str(nn_batch_size))
batch_size_file.close()


# set tensorflow seed
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(dt_int)  # set random seed for neural net model

# set philox state (Max accuracy state: [6, 6, 5])
dt_array = [int(i) for i in str(dt_int)]
random_state = dt_array[-3:]


# SCALE AND SPLIT DATA
X=data.iloc[:,:-1]

# split into train and test sets.
label = data['Outcome']
encoder = LabelEncoder()
X = data
y = label


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=dt_int)

# Write seed to file
seed_file = open(r"seed_file.txt", "w")
seed_file.write(str(dt_int))
seed_file.close()

# Import keras packages for neural network
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import random

# Set seed and random number generator for neural net
tf.random.Generator(state=random_state, alg="philox")

# write initial state to file
state_file = open(r"state_file.txt", "w")
state_file.write(str(random_state))
state_file.close()

# Build neural network and add layers
neural_net_model = Sequential()
neural_net_model.add(Dense(5, kernel_initializer='uniform', activation='relu'))
neural_net_model.add(Dense(26, kernel_initializer='uniform', activation='relu'))
neural_net_model.add(Dense(5, kernel_initializer='uniform', activation='relu'))
neural_net_model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# compile and fit model
neural_net_model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=['accuracy'])
K.set_value(neural_net_model.optimizer.learning_rate, 0.01)
neural_net_model.fit(X_train, y_train, batch_size=nn_batch_size, epochs=400, validation_data=(X_test, y_test))


# Evaluate neural net
from sklearn.metrics import classification_report, accuracy_score, f1_score
y_pred = neural_net_model.predict(X_test) > .9

out_file = open("out.txt", "w")
out_file.write("Accuracy: " + str(accuracy_score(y_test, y_pred)))
out_file.close()
