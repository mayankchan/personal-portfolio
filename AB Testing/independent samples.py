# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 08:23:07 2026

@author: HOMELC009452
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind,norm



#mock data

sample_A = norm.rvs(loc = 500, scale = 100, size = 250, random_state = 42).astype(int)

sample_B = norm.rvs(loc = 550, scale = 150, size = 100, random_state = 42).astype(int)


plt.hist(sample_A, density = True,alpha = 0.5)
plt.hist(sample_B, density = True,alpha = 0.5)
plt.show()


sample_A_mean = sample_A.mean()

sample_B_mean = sample_B.mean()

print(sample_A_mean,sample_B_mean)

#Set hypithesis


null_hypothesis = "The mean of the sample A is equal to the mean of the sample B"

alternate_hypothesis = "The mean of the sample A is differentl to the mean of the sample B"
acceptance_criteria = 0.05



#execute hypothesis


t_statistic, p_value = ttest_ind(sample_A, sample_B)

print(t_statistic, p_value)


#to reject null hypothesis: t-statistic should be above critical value and p value should be lower than acceptance criteria


#WELCH'S T-TEST ---> This works better when sample variances are different

t_statistic, p_value = ttest_ind(sample_A, sample_B,equal_var = False)

print(t_statistic, p_value)


