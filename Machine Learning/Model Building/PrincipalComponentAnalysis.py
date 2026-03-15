# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 05:10:46 2026

@author: HOMELC009452
"""

"""
PCA - Principal Component Analysis

Before applying PCA - 

1) To always apply scaling to the variables and use standardization and not normalisation

2) You will lose some of the information/variance contained in the original data

3) It will be much harder to interpret the outputs based on component values versus the original variables

"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.feature_selection import RFECV not needed for decision tree

data_for_model = pd.read_csv("data/sample_data_pca.csv")

# drop customer_id, not needed for modelling

data_for_model.drop("user_id", axis = 1, inplace = True) 

# Shuffle Data- To remove any ordering

data_for_model = shuffle(data_for_model,random_state = 42)


 # Class Balance
 
data_for_model["purchased_album"].value_counts()
data_for_model["purchased_album"].value_counts(normalize = True) # to check the %
 


# Missing Values

data_for_model.isna().sum()
data_for_model.isna().sum().sum() # this will give total missing across whole dataset 
data_for_model.dropna(how = "any",inplace = True) #Remove a row where any of the column is null



# Split Input and Output variables


X = data_for_model.drop("purchased_album", axis = 1)
y = data_for_model["purchased_album"]


# Split out Training and Test

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify = y)


# Feature Scaling - VERY IMP!!!

scale_standard = StandardScaler()
X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)


# Apply PCA

pca = PCA(n_components = None, random_state = 42)
pca.fit(X_train)

# Extract the explained variance across components

explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()




# Plot


num_vars_list = list(range(1,101))

plt.figure(figsize=(15,10)) # to increase the plot size since 2 plots are involved here


# plot the variance explained by each component

plt.subplot(2,1,1)
plt.bar(num_vars_list, explained_variance)
plt.title("Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("% Variance")
plt.tight_layout()

# plot the cumulative variance

plt.subplot(2,1,2)
plt.plot(num_vars_list, explained_variance_cumulative)
plt.title("Cumulative Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative % Variance")
plt.tight_layout()
plt.show()



# Reinstantiate PCA with known information


pca = PCA(n_components = 0.75, random_state = 42) # 0.75 is % variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# No. of components
pca.n_components_ 




clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train,y_train)


y_pred_class = clf.predict(X_test)
accuracy_score(y_test,y_pred_class)


"""
PCA components by definition are not co-related and could be extremely helpful in linear and logistic regression
Also helpful in clustering


"""




