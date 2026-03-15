# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 08:31:38 2026

@author: HOMELC009452
"""

#PairedT-Test sample result before and after an event



import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel,norm



#mock data

before = norm.rvs(loc = 500, scale = 100, size = 100, random_state = 42).astype(int)

np.random.seed(42)
after = before + np.random.randint(low = -50, high = 75, size = 100) #after event how much is the impact, added rand int for that to beofre set


plt.hist(before, density = True,alpha = 0.5, label = "Before")
plt.hist(after, density = True,alpha = 0.5, label = "After")
plt.legend()
plt.show()


before_mean = before.mean()

after_mean = after.mean()

print(before_mean,after_mean)

#Set hypithesis


null_hypothesis = "The mean of the before sample is equal to the mean of the after sample "

alternate_hypothesis = "The mean of the before sample A is differentl to the mean of the after sample "
acceptance_criteria = 0.05



#execute hypothesis


t_statistic, p_value = ttest_rel(before, after)

print(t_statistic, p_value)


#to reject null hypothesis: t-statistic should be above critical value and p value should be lower than acceptance criteria
