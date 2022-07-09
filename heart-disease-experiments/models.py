from genericpath import isfile
from random import Random
import sys
import os
from os.path import exists
import datetime
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier


# Set random number seed and write to file
if os.path.isfile("seedFile.txt"):
    # If seed file already exists, read from file
    seed_file = open("seedFile.txt", "r")
    seed = int(seed_file.read())
elif not os.path.isfile("seedFile.txt") or os.stat("seedFile.txt").st_size==0:
    # If no seed file exists, generate new seed from time method
    seed = int(datetime.utcnow().timestamp())
    np.random.seed(seed)  # Set global numpy seed

    # Create seed file and write to file
    seed_file = open("seedFile.txt", "w")
    seed_file.write(str(seed))

seed_file.close()  # Close seed file


# Get data set filepath from command line arguments
algorithm_type = sys.argv[1]
dataset_filename = sys.argv[2]

# Load and pre-process data
df = pd.read_csv(dataset_filename, header=None, na_values = {'?'})  
df.fillna(0, inplace=True)  # Fill NaN values

 # Split predictor and target columns
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)  

if algorithm_type=="gaussian-naive-bayes":
    ####### Gaussian Naive Bayes
    model = GaussianNB()
elif algorithm_type=="bernoulli-naive-bayes":
    ####### Bernoulli Naive Bayes
    model = BernoulliNB()
elif algorithm_type=="random-forest":
    ####### Random Forest
    model = RandomForestClassifier(random_state=seed)

# Fit and test model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# Write accuracy to out file
out_file = open("out.txt", "w")
out_file.write("Accuracy: " + str(model_accuracy))
out_file.close()