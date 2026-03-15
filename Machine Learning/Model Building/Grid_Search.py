# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 15:23:20 2026

@author: HOMELC009452
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV


my_df = pd.read_csv("data/sample_data_regression.csv")


X = my_df.drop(["output"], axis = 1)
y = my_df["output"]


# Instantiate GridSearch

gscv = GridSearchCV(
    estimator = RandomForestRegressor(random_state = 42),
    param_grid = {"n_estimators" : [10,50,100,500],
                  "max_depth" : [1,2,3,4,5,6,7,8,9,10,None]},
    cv = 5,
    scoring = "r2",
    n_jobs = -1 # to ensure all processors of system are utilized
    )



gscv.fit(X,y)

# Get the best CV score (mean)

gscv.best_score_

# What were the parameters (Optimal)

gscv.best_params_

regressor = gscv.best_estimator_ # Assign best estimator to regressor