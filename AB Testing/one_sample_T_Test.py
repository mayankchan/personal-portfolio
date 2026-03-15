# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 08:12:44 2026

@author: HOMELC009452
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_1samp,norm



#mock data

population = norm.rvs(loc = 500, scale = 100, size = 1000, random_state = 42).astype(int)

np.random.seed(42)
sample = np.random.choice(population, 250)


plt.hist(population, density = True,alpha = 0.5)
plt.hist(sample, density = True,alpha = 0.5)
plt.show()


population_mean = population.mean()

sample_mean = sample.mean()


#Set hypithesis


null_hypothesis = "The mean of the sample is equal to the mean of the population"

alternate_hypothesis = "The mean of the sample is differentl to the mean of the population"
acceptance_criteria = 0.05



#execute hypothesis


t_statistic, p_value = ttest_1samp(sample, population_mean)

print(t_statistic, p_value)


#to reject null hypothesis: t-statistic should be above critical value and p value should be lower than acceptance criteria


