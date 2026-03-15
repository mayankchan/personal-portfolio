# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 06:01:17 2026

@author: HOMELC009452
"""



# Balanced Data - ROC and Accuracy
# Imbalanced Data - Precision and Recall and F1 score




import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score,f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
# from sklearn.feature_selection import RFECV not needed for decision tree

data_for_model = pd.read_pickle("data/abc_classification_modelling.p")

data_for_model.drop("customer_id", axis = 1, inplace = True) # drop customer_id, not needed for modelling

# Shuffle Data- To remove any ordering

data_for_model = shuffle(data_for_model,random_state = 42)


 # Class Balance
 
data_for_model["signup_flag"].value_counts()
data_for_model["signup_flag"].value_counts(normalize = True) # to check the %
 


# Missing Values

data_for_model.isna().sum()
data_for_model.dropna(how = "any",inplace = True) #Remove a row where any of the column is null

# # Deal with Outliers

# outlier_investigation = data_for_model.describe()

# outlier_columns = ["distance_from_store","total_sales","total_items"]

# # 1st Approach - Boxplot approach

# for column in outlier_columns:
    
#     lower_quartile = data_for_model[column].quantile(0.25)
#     upper_quartile = data_for_model[column].quantile(0.75)
#     iqr = upper_quartile - lower_quartile
#     iqr_extended = iqr * 2
#     min_border = lower_quartile - iqr_extended
#     max_border = upper_quartile + iqr_extended
    
#     outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
#     print(f"{len(outliers)} detected in {column} column ")
    
#     data_for_model.drop(outliers, inplace = True)
    
# we would see that no outliers were detected in total_item, one reason would be that their outliers would have been removed
# in prior loops (distance_from_store and total_sales)


# Split Input and Output variables


X = data_for_model.drop("signup_flag", axis = 1)
y = data_for_model["signup_flag"]


# Split out Training and Test

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify = y)


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


# Feature Selection Important for logistic regression

# clf = LogisticRegression(random_state=42,max_iter= 1000) #no. of iterations to find the optimum regression line
# feature_selector = RFECV(clf) # we can also specify cross validation(CV) parameter, default =5, which splits data in parts

# fit = feature_selector.fit(X_train,y_train)


# optimal_feature_count = feature_selector.n_features_ #tells what is the optimal no. of input variables

# print(f"{optimal_feature_count} is the optimal number of features")

# X_train = X_train.loc[:, feature_selector.get_support()]
# X_test = X_test.loc[:, feature_selector.get_support()]



# plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
# plt.ylabel("Model Score")
# plt.xlabel("Number of Features")
# plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
# plt.tight_layout()
# plt.show()


# Model Training

clf = RandomForestClassifier(random_state=42, n_estimators = 500, max_features = 5)
clf.fit(X_train, y_train)


# Model Assessment


y_pred_class = clf.predict(X_test)

# To fetch probabilities of the output
y_pred_prob = clf.predict_proba(X_test)[:,1]


# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)


plt.style.available

plt.style.use('seaborn-v0_8-poster')
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")

for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i,corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()



# Accuracy ( the number of correct classification out of all attempted classifications)
accuracy_score(y_test,y_pred_class)


# Precision ( of all obseravations that were predicted as positive, how many were actually positive)
precision_score(y_test,y_pred_class)


# Recall ( of all positive obseravations, how many did we predict as positive) TPR Tur Positive Rate
recall_score(y_test,y_pred_class)

# F1-Score ( the harmonic mean of Recall and Precision)
f1_score(y_test,y_pred_class)


# # Findig the best max depth

# max_depth_list = list(range(1,15))

# accuracy_scores =[]

# for depth in max_depth_list:
    
#     clf = DecisionTreeClassifier(max_depth= depth, random_state=  42)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = f1_score(y_test,y_pred)
#     accuracy_scores.append(accuracy)

# max_accuracy = max(accuracy_scores)
# max_accuracy_idx = accuracy_scores.index(max_accuracy)
# optimal_depth = max_depth_list[max_accuracy_idx]


# Feature Importance (mean decrease in gini co-efficient of impurity)



feature_importance = pd.DataFrame(clf.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)


plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Permutation Importance - considered slightly superior than feature importance
# decrease seen when we randomize the values of one input variable 


result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state = 42)


permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)


plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

"""
if permutation is showing negative then it means that variable importance is more pronounced on shuffled data
than it is on real data, it means that the variable is not of much importance

Here, we can see that case with total_sales and gender

"""





