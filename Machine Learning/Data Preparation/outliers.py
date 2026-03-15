# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 19:37:33 2026

@author: HOMELC009452
"""

"""
 Two common approaches for identifying outliers
 
 1) box-plot to identify percentiles (25,75) and multiplying range by 1.5(commonly used but not fixed) to identify min and max

 2) -3SD to 3SD --> 99.7% using normal distribution, anything outside will fall into outliers

"""

 
import pandas as pd


my_df = pd.DataFrame({"input1" : [15,41,44,47,50,53,56,59,99],
                      "input2" : [29,41,44,47,50,53,56,59,66],
                      })

my_df.plot(kind = "box", vert = False)# vert false means plot will be horizontal


outlier_columns = ["input1","input2"]

# 1st Approach - Boxplot approach

for column in outlier_columns:
    
    lower_quartile = my_df[column].quantile(0.25)
    upper_quartile = my_df[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 1.5
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} detected in {column} column ")
    
    my_df.drop(outliers, inplace = True)
    
    
# 2nd Approach - Standard Deviation approach

my_df = pd.DataFrame({"input1" : [15,41,44,47,50,53,56,59,99],
                      "input2" : [29,41,44,47,50,53,56,59,66],
                      })

my_df.plot(kind = "box", vert = False)# vert false means plot will be horizontal


outlier_columns = ["input1","input2"]



for column in outlier_columns:
    
    mean = my_df[column].mean()
    std_dev = my_df[column].std()
    
    min_border = mean - std_dev * 3
    max_border = mean + std_dev * 3
    
    outliers = my_df[(my_df[column] < min_border) | (my_df[column] > max_border)].index
    print(f"{len(outliers)} detected in {column} column ")
    
    my_df.drop(outliers, inplace = True)
