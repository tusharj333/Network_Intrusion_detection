# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
#%%

network_data = pd.read_csv('network.csv', header = None, delimiter = ' *, *')

network_data.head()

#%% Assigning colomn names
network_data.columns=["Node","Utilised Bandwith Rate","Packet Drop Rate","Full_Bandwidth","Average_Delay_Time_Per_Sec",
"Percentage_Of_Lost_Pcaket_Rate","Percentage_Of_Lost_Byte_Rate","Packet Received Rate","of Used_Bandwidth",
"Lost_Bandwidth","Packet Size_Byte","Packet_Transmitted","Packet_Received","Packet_lost","Transmitted_Byte",
"Received_Byte","10-Run-AVG-Drop-Rate","10-Run-AVG-Bandwith-Use","10-Run-Delay","Node Status","Flood Status","Class"]

network_data.head()
#%%
print(network_data.describe(include='all'))
print(network_data.isnull().sum())

#%%
pd.set_option("display.max_columns", None)

# create a copy of Data Frame
network_data_rev = pd.DataFrame.copy(network_data)

#%%
# label encoding 
col = ["Node", "Full_Bandwidth","Packet Size_Byte", "Node Status", "Class" ]

from sklearn import preprocessing

le = {}

for x in col:
    le[x] = preprocessing.LabelEncoder()
for x in col:
    network_data_rev[x] = le[x].fit_transform(network_data_rev.__getattr__(x))
    
network_data_rev.head()

#%% Split X and Y

X = network_data_rev.values[:, :-1]
Y = network_data_rev.values[:,-1]

#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Y = Y.astype(int)
#%% Training and testing data 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 10) 

#%%  Decision Tree
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree = DecisionTreeClassifier()

model_DecisionTree.fit(X_train, Y_train)
Y_pred = model_DecisionTree.predict(X_test)  

#%% 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred))
print("Accuracy Score : ", accuracy_score(Y_test, Y_pred))
print("Classification Report: \n", classification_report(Y_test, Y_pred))

#%% Cross validation
model = DecisionTreeClassifier()
from sklearn import cross_validation
kfold_cv = cross_validation.KFold(n = len(X_train), n_folds = 10)
kfold_cv_results = cross_validation.cross_val_score(estimator = model,X=X_train, y=Y_train, scoring =  "accuracy", cv = kfold_cv)
print(kfold_cv_results.mean())

#%% Logistic Regression 
from sklearn.linear_model import LogisticRegression

model_Log_reg = LogisticRegression()
model_Log_reg.fit(X_train, Y_train)
Y_pred_lg = model_Log_reg.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred_lg))
print("Accuracy Score : ", accuracy_score(Y_test, Y_pred_lg))
print("Classification Report: \n", classification_report(Y_test, Y_pred_lg))

#%% SVM
from sklearn import svm

svc_model = svm.SVC(kernel = 'rbf', C = 1.0, gamma = 0.1)
svc_model.fit(X_train, Y_train)
Y_pred_svm = svc_model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred_svm))
print("Accuracy Score : ", accuracy_score(Y_test, Y_pred_svm))
print("Classification Report: \n", classification_report(Y_test, Y_pred_svm))

#%%
