# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 04:57:09 2026

@author: HOMELC009452
"""

# Import packages

from causalimpact import CausalImpact
import pandas as pd


# import data tables

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
campaign_data = pd.read_excel("data/grocery_database.xlsx", sheet_name = "campaign_data")


# aggregate - ALWAYS USE RESET_INDEX()

customer_daily_sales = transactions.groupby(["customer_id","transaction_date"])["sales_cost"].sum().reset_index()

# merger on the signup flag

customer_daily_sales = pd.merge(customer_daily_sales,campaign_data, how="inner", on ="customer_id")


# Pivot data to transaction_date

causal_impact_df = customer_daily_sales.pivot_table(index="transaction_date",
                                                    columns = "signup_flag",
                                                    values = "sales_cost",
                                                    aggfunc = "mean")

# Frequency will give warning message, but causal will still run

causal_impact_df.index

# change it to D, will remove warning

causal_impact_df.index.freq = "D"


# for causal impact we need the impacted group in the first column, i.e. member

causal_impact_df = causal_impact_df[[1,0]]

causal_impact_df.columns = ["member","non_member"]

# non member is the set utilized for counterfactual projection

# Apply Causal Impact


pre_period = ["2020-04-01","2020-06-30"]
post_period = ["2020-07-01","2020-09-30"]


ci = CausalImpact(causal_impact_df,pre_period,post_period)

ci.plot()




# Extract the summary

print(ci.summary())

print(ci.summary(output = "report"))