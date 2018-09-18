#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:01:17 2018

@author: standarduser
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values


# elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++',n_init=10,random_state=0,max_iter=300)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.show()

# applying kmeans to mall dataset
kmeans = KMeans(n_clusters=5,random_state=0,init='k-means++',n_init=10)
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[y_kmeans == 0,1], X[y_kmeans == 0,0],s = 100, c= 'red',label= 'Cluster1')
plt.scatter(X[y_kmeans == 1,1], X[y_kmeans == 1,0],s = 100, c= 'blue',label= 'Cluster2')
plt.scatter(X[y_kmeans == 2,1], X[y_kmeans == 2,0],s = 100, c= 'green',label= 'Cluster3')
plt.scatter(X[y_kmeans == 3,1], X[y_kmeans == 3,0],s = 100, c= 'cyan',label= 'Cluster4')
plt.scatter(X[y_kmeans == 4,1], X[y_kmeans == 4,0],s = 100, c= 'magenta',label= 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label = 'Centroids')
plt.legend()
plt.show()



