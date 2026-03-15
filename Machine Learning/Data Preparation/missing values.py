# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 12:19:08 2026

@author: HOMELC009452
"""

import pandas as pd
import numpy as np

my_df = pd.DataFrame({"A" : [1,2,4,np.nan,5,np.nan,7],
                      "B" : [4,np.nan,7,np.nan,1,np.nan,2]})

#Finding missing values with Pandas

my_df.isna().sum() #denote how many missing values are there per column


# Dropping Missing Values with Pandas

my_df.dropna()


my_df.dropna(how = "any") #default setting for dropna function, if any column is missing value, then whole row of data is dropped

#help functionality for any function - press ctrl + i


my_df.dropna(how = "all")


my_df.dropna(how = "any", subset = ["A"]) # will only check for column A and then drop whole of the row

my_df.dropna(how = "any", subset = ["A"],inplace = True) # inplace will change the DataFrame



#Filling Missing Values with Pandas


my_df.fillna(value = 100) # fill constant value 100 for any missing value in the DataFrame - not recommended



mean_value = my_df["A"].mean() 
my_df["A"].fillna(value = mean_value)

my_df.fillna(value = my_df.mean(), inplace = True) # both the columns missing values wiil be filled by their respective means



#Missing values with Simple Imputer using scikit learn

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


my_df = pd.DataFrame({"A": [1,4,7,10,13],
                      "B": [3,6,9,np.nan,15],
                      "C": [2,5,np.nan,11,np.nan]})


imputer = SimpleImputer()


imputer.fit(my_df) # model is trained, so use training data for this
imputer.transform(my_df)

my_df1 = imputer.transform(my_df) # sci-kit returns an array

my_df2 = pd.DataFrame(my_df1,columns = my_df.columns)


imputer.fit_transform(my_df)    #only apply this on training data


my_df["B"]=imputer.fit_transform(my_df[["B"]]) #ML only applied to column B, model will learn based on B only, and will apply values there only


#KNNImputer

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


my_df = pd.DataFrame({"A": [1,2,3,4,5],
                      "B": [1,1,3,3,4],
                      "C": [1,2,9,np.nan,20]})

#Simmple Imputer would have only looked at Column C mean for missing value there, KNN has the option to look at A, B co-ordinates



knn_imputer= KNNImputer() # default: all are considered neighbors and mean is calculated
knn_imputer= KNNImputer(n_neighbors = 1)
knn_imputer= KNNImputer(n_neighbors = 2)
knn_imputer= KNNImputer(n_neighbors = 2, weights = "distance")
knn_imputer.fit_transform(my_df) 


my_df1 = pd.DataFrame(knn_imputer.fit_transform(my_df) , columns = my_df.columns) # convert array returned object by model into DF














































