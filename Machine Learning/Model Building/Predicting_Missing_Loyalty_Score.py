# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 18:21:07 2026

@author: HOMELC009452
"""

import pandas as pd
import pickle


# Import customers for which scoring is required to be calculated

to_be_scored = pickle.load(open("data/abc_regression_scoring.p", "rb"))


# Import model and model objects

regressor = pickle.load(open("data/random_forest_regression_model.p", "rb"))
one_hot_encoder = pickle.load(open("data/random_forest_regression_ohe.p", "rb"))


# Drop unsed columns

to_be_scored.drop("customer_id", axis = 1, inplace = True)

# Drop missing values

to_be_scored.dropna(how = "any", inplace = True)


# Apply one hot encoding

categorical_vars = ["gender"]
encoded_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars]) # no fit, since we are directly utilzing existing one_hot_encoder pickle object from earlier model based on training data
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)
encoder_vars_df = pd.DataFrame(encoded_vars_array,columns = encoder_feature_names)
to_be_scored = pd.concat([to_be_scored.reset_index(drop = True),encoder_vars_df.reset_index(drop = True)], axis = 1)
to_be_scored.drop(categorical_vars,axis = 1, inplace = True)


# Make our predictions !

loyalty_predictions = regressor.predict(to_be_scored)





