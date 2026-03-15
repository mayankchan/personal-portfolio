# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 06:07:00 2026

@author: HOMELC009452
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd



my_df = pd.read_csv("data/sample_data_regression.csv")


# split data

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]



X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



# Instantiate our model object


regressor = DecisionTreeRegressor()


# Train our model

regressor.fit(X_train, y_train)

# Assess model accuracy

y_pred = regressor.predict(X_test)

print(r2_score(y_test,y_pred))


# A Demonstration of overfitting - A decision tree can be very prone to that

y_pred_training = regressor.predict(X_train)

print(r2_score(y_train,y_pred_training)) # it comes out 1 which shows that it has perfectly modelled, rather overfitted, based on inputs provided



import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = list(X.columns),
                 filled = True, 
                 rounded = True,
                 fontsize = 24
                 )



# To remedy overfitting, let's introduce a min_samples_leaf parameter


regressor = DecisionTreeRegressor(min_samples_leaf = 7)


# Train our model

regressor.fit(X_train, y_train)

# Assess model accuracy

y_pred = regressor.predict(X_test)

print(r2_score(y_test,y_pred))

y_pred_training = regressor.predict(X_train)

r2_score(y_train,y_pred_training)

# Plot the model tree

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = list(X.columns),
                 filled = True, 
                 rounded = True,
                 fontsize = 24
                 )
