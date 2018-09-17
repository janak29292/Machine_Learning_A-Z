# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:46:58 2018

@author: Janak
"""

import numpy
import pandas
from matplotlib import pyplot
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.formula import api

#importing dataset and distinguishing independent and dependent variables
dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values


# fitting linear regression model to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# fitting polynomial regression model to the dataset
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 =LinearRegression()
lin_reg2.fit(X_poly, Y)

# visualising linear regression results
pyplot.scatter(X, Y, color = 'red')
pyplot.plot(X, lin_reg.predict(X), color = 'blue')
pyplot.title('linear model for salary prediction')
pyplot.xlabel('Position')
pyplot.ylabel('salary')
pyplot.show()

# visualising polynomial regression results
X_grid = numpy.arange(min(X), max(X), 0.1)
X_grid = numpy.reshape(X_grid,(len(X_grid), 1 ))

pyplot.scatter(X, Y, color = 'red')
pyplot.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
pyplot.title('polynomial model for salary prediction')
pyplot.xlabel('Position')
pyplot.ylabel('salary')
pyplot.show()


# prediction using linear model
lin_reg.predict(6.5)

# prediction using polynomial model
lin_reg2.predict(poly_reg.fit_transform(6.5))