# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 05:02:36 2026

@author: HOMELC009452
"""

"""
Two ways for feature scaling

1) standardization - Rescales data to have a mean of 0 and deviation of 1
                     liner and logistic regression, intensity of outliers matters
                     
                     
2) normalization - rescales data so that it exists in a range of 0 and 1 , by min and max calculation
                    if you want data to be +ve - normalization, like image data and categorical 
                    

Models where distance matters like , linear and logistic , Kmeans - scaling is important and help with faster execution rather than accuracy
whereas in cases like decision tree model and random forest where each variable is handled independently, scaling doesn't matter much

"""


 
import pandas as pd


my_df = pd.DataFrame({"height" : [1.98,1.77,1.76,1.80,1.64],
                      "weight" : [99,81,70,86,82]
                      })

# Standardization

from sklearn.preprocessing import StandardScaler


scale_standard = StandardScaler()
scale_standard.fit_transform(my_df)
#scale_standard.fit_transform(my_df[["Height"]]) # if we want model to work only on on column

my_df_standardized = pd.DataFrame(scale_standard.fit_transform(my_df), columns = my_df.columns)


# Normalization

from sklearn.preprocessing import MinMaxScaler


min_max_scaler = MinMaxScaler()
min_max_scaler.fit_transform(my_df)

my_df_normalized = pd.DataFrame(min_max_scaler.fit_transform(my_df), columns = my_df.columns)