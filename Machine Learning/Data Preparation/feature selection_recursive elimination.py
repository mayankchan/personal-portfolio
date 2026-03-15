# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 13:23:31 2026

@author: HOMELC009452
"""


# recursive feature elimination with cross validation

import pandas as pd

my_df = pd.read_csv("feature_selection_sample_data.csv")

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression


X = my_df.drop(["output"],axis = 1)
y = my_df["output"]


regressor = LinearRegression()
feature_selector = RFECV(regressor) # we can also specify cross validation(CV) parameter, default =5, which splits data in parts

fit = feature_selector.fit(X,y)


optimal_feature_count = feature_selector.n_features_ #tells what is the optimal no. of input variables

print(f"{optimal_feature_count} is the optimal number of features")

X_new = X.loc[:, feature_selector.get_support()]


import matplotlib.pyplot as plt

plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()


