#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:11:23 2023

@author: nebiyousamuel
"""

import pandas as pd
import statsmodels.api as sm

# Load the dataset from the CSV file
data = pd.read_csv("toyDataSet.csv")

# Split the data into independent variable (X) and dependent variable (Y)
X = data["X"]
y = data["Y"]

# Add constant term to the independent variable for the intercept term in the model
X = sm.add_constant(X)

# Create the OLS model and fit it
model = sm.OLS(y, X)
results = model.fit()

# Print the summary to evaluate the fit
print(results.summary())
