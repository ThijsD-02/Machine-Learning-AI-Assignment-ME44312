import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
from collections import Counter
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

db = pd.read_csv('data/modes.csv')

X = db[['xmin', 'ymin', 'zmin', 'xmean', 'ymean', 'zmean', 'xstd', 'ystd', 'zstd']].values
X

y = db['Class']
#label encoding is done as model accepts only numeric values
# so strings need to be converted into labels
LE = preprocessing.LabelEncoder()
LE.fit(y)
y = LE.transform(y)
y

#splitting dataset into train, validation and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,random_state = 0)

# Scale data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
train_logreg = LogisticRegression(random_state=0, max_iter=200).fit(X_train, y_train)

# Train a Neural Network (MLPClassifier)
train_nn = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=0)
train_nn.fit(X_train, y_train)

# Predictions for Logistic Regression
pred_logreg = train_logreg.predict(X_val)
print("For Logistic Regression: ")
print(classification_report(y_val, pred_logreg))
print("Accuracy of logistic regression on the initial data is: ", accuracy_score(y_val, pred_logreg))

# Predictions for Neural Network
pred_nn = train_nn.predict(X_val)
print("For Neural Network: ")
print(classification_report(y_val, pred_nn))
print("Accuracy of NN on the initial data is: ", accuracy_score(y_val, pred_nn))
