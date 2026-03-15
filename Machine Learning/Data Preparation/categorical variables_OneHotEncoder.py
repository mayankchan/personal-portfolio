# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 18:56:11 2026

@author: HOMELC009452
"""

#Dummy variable trap - introduce n-1 dummy variables to eliminate multicollinearity among columns


import pandas as pd
from sklearn.preprocessing import OneHotEncoder



X = pd.DataFrame({"input1" : [1,2,3,4,5],
                  "input2" : ["A","A","B","B","C"],
                  "input3" : ["X","X","X","Y","Y"]})


categorical_vars = ["input2","input3"]


one_hot_encoder = OneHotEncoder(sparse_output = False, drop = "first") #returned an array rather than sparse object, drop ensures that dummy variable trap is handled by dropping one of the created columns

encoded_vars_array =one_hot_encoder.fit_transform(X[categorical_vars])


encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)



encoder_vars_df = pd.DataFrame(encoded_vars_array,columns = encoder_feature_names)

X_new = pd.concat([X.reset_index(drop = True),encoder_vars_df.reset_index(drop = True)], axis = 1)


X_new.drop(categorical_vars,axis = 1, inplace = True)




