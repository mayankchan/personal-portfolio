# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 11:41:28 2026

@author: HOMELC009452
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



my_df = pd.read_csv("data/sample_data_classification.csv")


# split data

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]


# stratify is to ensure that training and test set data contains the same proportion of 1s and 0s (classification)
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)



# Instantiate our model object


clf = LogisticRegression(random_state = 42)


# Train our model

clf.fit(X_train, y_train)

# Assess model accuracy

y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred)


# To fetch probabilities of the output
y_pred_prod = clf.predict_proba(X_test)


# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.style.available

plt.style.use('seaborn-v0_8-poster')
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")

for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i,corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()



