"""
Created on Tue Sep 18 16:41:24 2018

@author: standarduser
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)

transaction=[[str(b) for b in a] for a in dataset.values]

# training apriori on dataset
from apyori import apriori
rules = apriori(transaction,min_support=0.0028,min_confidence=0.2,min_lift=3,min_length=2)

# min_support: 3*7/7500 i.e. minimum of 3 times a day for a week of transactions i.e. 7500 transactions

# visualising the results
results = list(rules)