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

#Feature Extraction from Timestamp column
db['Start Time'] = db['Start Time'].astype('datetime64[ns]')
db['hour'] = db['Start Time'].dt.hour
db['minute'] = db['Start Time'].dt.minute
db['day'] = db['Start Time'].dt.day
print(db.head())

#Again training the classifiers
X = db.iloc[:,4:]
y = db['Class']
#label encoding is done as model accepts only numeric values
# so strings need to be converted into labels
LE = preprocessing.LabelEncoder()
LE.fit(y)
y = LE.transform(y)

#splitting dataset into train, validation and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,random_state = 0)

# Scale data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert the y variable into one-hot encoding - basically the true label will be 1 and all others will be assigned to 0
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_oh = one_hot(y_train, len(np.unique(y)))
y_val_oh = one_hot(y_val, len(np.unique(y)))
y_test_oh = one_hot(y_test, len(np.unique(y)))

train_logreg2 = LogisticRegression(random_state=0,max_iter = 200).fit(X_train,y_train)

# Train a Neural Network (MLPClassifier)
train_nn2 = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=0)
train_nn2.fit(X_train, y_train)

pred_logreg2 = train_logreg2.predict(X_val)
print("For Logistic Regression: ")
print(classification_report(y_val, pred_logreg2))
print ("Accuracy of logistic regression on the extended data is: ",accuracy_score(pred_logreg2,y_val)*100,'%')

# Predictions for Neural Network
pred_nn2 = train_nn2.predict(X_val)
print("For Neural Network: ")
print(classification_report(y_val, pred_nn2))
print("Accuracy of NN on the initial data is: ", accuracy_score(y_val, pred_nn2)*100,'%')

#Accuracy of the models should increase by using additional features
#Pick the one with the highest accuracy and apply it to the test data. 

# Apply the chosen model to test data
final_res = train_nn2.predict(X_test)

# Compute and print final accuracy
final_accuracy = accuracy_score(y_test, final_res)
print("Accuracy of the chosen Classifier on the test data is: ", final_accuracy * 100 ,'%')

# Ensure accuracy is at least 80%, otherwise, revisit model
if final_accuracy < 0.80:
    print("Warning: Accuracy is below 80%. Consider improving feature engineering or tuning hyperparameters.")