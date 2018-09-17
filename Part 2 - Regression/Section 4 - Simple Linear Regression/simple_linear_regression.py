# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 19:24:31 2018

@author: Janak
"""

import numpy
import pandas
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing dataset and distinguishing independent and dependent variables
dataset = pandas.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# splitting dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0) # dont (need to) put random_state in your code


# feature scaling
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_train)

#fitting simple linear regression to training set
regsr = LinearRegression()
regsr.fit(X_train, Y_train)

# predecting the test set result
Y_pred = regsr.predict(X_test)

#visualising the training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regsr.predict(X_train), color='blue')
plt.title('salary vs experience (training set)')
plt.xlabel('years of xp')
plt.ylabel('salary')
plt.show()


#visualising the test set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regsr.predict(X_train), color='blue')
plt.title('salary vs experience (test set)')
plt.xlabel('years of xp')
plt.ylabel('salary')
plt.show()