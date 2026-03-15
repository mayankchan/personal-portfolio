# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 05:54:31 2026

@author: HOMELC009452
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



my_df = pd.read_csv("data/sample_data_regression.csv")


# split data

X = my_df.drop(["output"], axis = 1)
y = my_df["output"]



X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



# Instantiate our model object


regressor = LinearRegression()


# Train our model

regressor.fit(X_train, y_train)

# Assess model accuracy

y_pred = regressor.predict(X_test)

r2_score(y_test,y_pred)