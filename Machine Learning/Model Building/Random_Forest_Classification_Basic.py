# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 05:39:30 2026

@author: HOMELC009452
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd



my_df = pd.read_csv("data/sample_data_classification.csv")


# split data

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]



X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify = y)



# Instantiate our model object


clf = RandomForestClassifier(random_state = 42)


# Train our model

clf.fit(X_train, y_train)

# Assess model accuracy

y_pred = clf.predict(X_test) # default threshold is 0.5

print(accuracy_score(y_test,y_pred))


# # A Demonstration of overfitting - A decision tree can be very prone to that

# y_pred_training = clf.predict(X_train)

# print(accuracy_score(y_train,y_pred_training)) # without min sample leaf, it would have been 1





