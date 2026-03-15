# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 05:27:26 2026

@author: HOMELC009452
"""

"""
Feature Selection is the process used to select the input variables that are most important to ML task

Why is it needed? 

a) Improved Model Accuracy
b) Far quicker and easier to learn (lower computation cost)
c) Easier to understand and explain to external stakeholders

Methods - 

a) Univariate Feature Selection

Applying statistical tests to find relationships between output variable, and each input variable in isolation


b) Recursive Feature Elimination

Fits a model that starts with all input variables, then iteratively removes those with the weakest relationship with the 
output until the desired number of features is reached


"""

# Correlation Matrix

import pandas as pd

my_df = pd.read_csv("feature_selection_sample_data.csv")

correlation_matrix = my_df.corr()



# Univariate Testing

import pandas as pd
my_df = pd.read_csv("feature_selection_sample_data.csv")


# Regression Template

from sklearn.feature_selection import SelectKBest, f_regression #f_regression is used for regression tasks

X = my_df.drop(["output"],axis=1)
y = my_df["output"]

feature_selector = SelectKBest(f_regression, k ="all")
fit = feature_selector.fit(X,y)

fit.pvalues_   #to check pscores, lower means more significant
fit.scores_    #to check f scores, higher means more significant

p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
input_variable_name = pd.DataFrame(X.columns)
summary_stats = pd.concat([input_variable_name,p_values,scores],axis =1)
summary_stats.columns = ["input_variable_name","p-value","f_scores"]

summary_stats.sort_values(by = "p-value", inplace = True)

p_value_threshold = 0.05
score_threshold = 5


selected_variables = summary_stats.loc[(summary_stats["p-value"] < p_value_threshold) & (summary_stats["f_scores"] > score_threshold) ]

selected_variables = selected_variables["input_variable_name"].to_list() #selected variables are put in list &
X_new = X[selected_variables] # above list is utilized to only fetch values of those columns and this will be fed to model as input


# IF we want a desirable no. of variables like 2

feature_selector = SelectKBest(f_regression, k =2)
fit = feature_selector.fit(X,y)

X_new1 = feature_selector.transform(X) # this directly provides array of values of selected variables

feature_selector.get_support() #to get list of column names True or False - depending upon which all columns were selected
X_new1 = X.loc[:, feature_selector.get_support()] #transformed with column names 


# Classification Template

from sklearn.feature_selection import SelectKBest, chi2 # chi2 is used for classification tasks

X = my_df.drop(["output"],axis=1)
y = my_df["output"]

feature_selector = SelectKBest(chi2, k ="all")
fit = feature_selector.fit(X,y)

fit.pvalues_   #to check pscores, lower means more significant
fit.scores_    #to check f scores, higher means more significant

p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)
input_variable_name = pd.DataFrame(X.columns)
summary_stats = pd.concat([input_variable_name,p_values,scores],axis =1)
summary_stats.columns = ["input_variable_name","p-value","chi2_score"]

summary_stats.sort_values(by = "p-value", inplace = True)

p_value_threshold = 0.05
score_threshold = 5


selected_variables = summary_stats.loc[(summary_stats["p-value"] < p_value_threshold) & (summary_stats["chi2_score"] > score_threshold) ]

selected_variables = selected_variables["input_variable_name"].to_list() #selected variables are put in list &
X_new = X[selected_variables] # above list is utilized to only fetch values of those columns and this will be fed to model as input






























