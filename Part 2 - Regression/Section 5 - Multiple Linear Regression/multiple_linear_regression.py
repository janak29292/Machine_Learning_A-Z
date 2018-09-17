# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 23:16:31 2018

@author: Janak
"""

import numpy
import pandas
from matplotlib import pyplot
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.formula import api

#importing dataset and distinguishing independent and dependent variables
dataset = pandas.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# encoding catagorical data
lencoder_X = LabelEncoder()
X[:,3] = lencoder_X.fit_transform(X[:,3])
hotcoder = OneHotEncoder(categorical_features = [3])
X = hotcoder.fit_transform(X).toarray()

# avoiding dummy variable trap
X = X[:,1:]  # don't need (included in LinearRegression model)

# splitting dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # dont (need to) put random_state in your code

# fit multiple linear regression model to training set

regrsr = LinearRegression()
regrsr.fit(X_train, Y_train)
Y_pred = regrsr.predict(X_test)


# building optimal model using backward elimination
X = numpy.append(arr = numpy.ones((50, 1)).astype(int), values = X, axis=1) # adding a column of 1 (constant) at the begining of matrix(required by statsmodel library)
X_opt  = X[:,[0,1,2,3,4,5]]
regrsr_ols = api.OLS(endog=Y, exog=X_opt).fit()
regrsr_ols.summary()

X_opt  = X_opt[:,[0,1,3,4,5]]
regrsr_ols = api.OLS(endog=Y, exog=X_opt).fit()
regrsr_ols.summary()

X_opt  = X_opt[:,[0,2,3,4]]
regrsr_ols = api.OLS(endog=Y, exog=X_opt).fit()
regrsr_ols.summary()

X_opt  = X_opt[:,[0,1,3]]
regrsr_ols = api.OLS(endog=Y, exog=X_opt).fit()
regrsr_ols.summary()

X_opt  = X_opt[:,[0,1]]
regrsr_ols = api.OLS(endog=Y, exog=X_opt).fit()
regrsr_ols.summary()

