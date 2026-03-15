# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 14:00:13 2026

@author: HOMELC009452
"""

import pandas as pd

my_df = pd.read_csv("feature_selection_sample_data.csv")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X = my_df.drop(["output"],axis = 1)
y = my_df["output"]


# Regression Model does not really need Stratify
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


# Classification Model need Stratify
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42,stratify = y)


regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
r2_score(y_test,y_pred)

# Cross validation

from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold

cv_scores = cross_val_score(regressor,X,y,cv = 4,scoring= "r2")
cv_scores.mean()

# One thing to note above, cross validation does not shuffle data and there is no provision of random seed. To overcome this, we make use of kfold

# Regression Template

cv = KFold(n_splits = 4, shuffle= True,random_state=42)
cv_scores = cross_val_score(regressor,X,y,cv = cv,scoring= "r2") #only change is that cv is a set of parameters rather than any number, 4 previously
cv_scores.mean()


# Classification Template

cv = StratifiedKFold(n_splits = 4, shuffle= True,random_state=42)
cv_scores = cross_val_score(clf,X,y,cv = cv,scoring= "accuracy") 
cv_scores.mean()




