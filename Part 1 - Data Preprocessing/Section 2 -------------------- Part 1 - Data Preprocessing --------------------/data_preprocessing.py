# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:51:13 2018

@author: Janak
"""
import numpy as np
import pandas
from matplotlib import pyplot
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#importing dataset and distinguishing independent and dependent variables
dataset = pandas.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#taking care of missing data
imputer = Imputer()
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding catagorical data
lencoder = LabelEncoder()
X[:,0] = lencoder.fit_transform(X[:,0])

hotcoder = OneHotEncoder(categorical_features = [0])
X = hotcoder.fit_transform(X).toarray()

lecodery = LabelEncoder()
Y = lecodery.fit_transform(Y)

# splitting dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # dont (need to) put random_state in your code


# feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_train)