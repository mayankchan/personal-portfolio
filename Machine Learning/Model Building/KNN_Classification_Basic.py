# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 06:42:16 2026

@author: HOMELC009452
"""




from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd



my_df = pd.read_csv("data/sample_data_classification.csv")


# split data

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]



X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify = y)



# Instantiate our model object


clf = KNeighborsClassifier() # there is no random_state since there is no randomness in this model


# Train our model

clf.fit(X_train, y_train)

# Assess model accuracy

y_pred = clf.predict(X_test) # default threshold is 0.5

print(accuracy_score(y_test,y_pred))







