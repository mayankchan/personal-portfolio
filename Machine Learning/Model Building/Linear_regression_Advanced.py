# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 03:36:21 2026

@author: HOMELC009452
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

data_for_model = pickle.load(open("data/abc_regression_modelling.p", "rb"))

data_for_model.drop("customer_id", axis = 1, inplace = True) # drop customer_id, not needed for modelling

# Shuffle Data- To remove any ordering

data_for_model = shuffle(data_for_model,random_state = 42)


# Missing Values

data_for_model.isna().sum()
data_for_model.dropna(how = "any",inplace = True) #Remove a row where any of the column is null

# Deal with Outliers

outlier_investigation = data_for_model.describe()

outlier_columns = ["distance_from_store","total_sales","total_items"]

# 1st Approach - Boxplot approach

for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} detected in {column} column ")
    
    data_for_model.drop(outliers, inplace = True)
    
# we would see that no outliers were detected in total_item, one reason would be that their outliers would have been removed
# in prior loops (distance_from_store and total_sales)


# Split Input and Output variables


X = data_for_model.drop("customer_loyalty_score", axis = 1)
y = data_for_model["customer_loyalty_score"]


# Split out Training and Test

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Deal with Categorical Variable


categorical_vars = ["gender"]


one_hot_encoder = OneHotEncoder(sparse_output = False, drop = "first") #returned an array rather than sparse object, drop ensures that dummy variable trap is handled by dropping one of the created columns

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])


encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)


X_train_encoded = pd.DataFrame(X_train_encoded,columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True),X_train_encoded.reset_index(drop = True)], axis = 1)
X_train.drop(categorical_vars,axis = 1, inplace = True)



X_test_encoded = pd.DataFrame(X_test_encoded,columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True),X_test_encoded.reset_index(drop = True)], axis = 1)
X_test.drop(categorical_vars,axis = 1, inplace = True)


# Feature Selection

regressor = LinearRegression()
feature_selector = RFECV(regressor) # we can also specify cross validation(CV) parameter, default =5, which splits data in parts

fit = feature_selector.fit(X_train,y_train)


optimal_feature_count = feature_selector.n_features_ #tells what is the optimal no. of input variables

print(f"{optimal_feature_count} is the optimal number of features")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]



plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()


# Model Training

regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Model Assessment

# Predict on the Test Set
y_pred = regressor.predict(X_test)


# Calculate R-Squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)


# Cross Validation
cv = KFold(n_splits = 4, shuffle= True,random_state=42)
cv_scores = cross_val_score(regressor,X_train, y_train,cv = cv,scoring= "r2") #only change is that cv is a set of parameters rather than any number, 4 previously
cv_scores.mean()

# Calculate Adjusted R-Squared
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars)
print(adjusted_r_squared)


# Extract Model Coefficients

coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names,coefficients], axis =1)
summary_stats.columns = ["input_variable", "coeffi"]



# Extract Model Intercept ( y = mx + c)

regressor.intercept_
