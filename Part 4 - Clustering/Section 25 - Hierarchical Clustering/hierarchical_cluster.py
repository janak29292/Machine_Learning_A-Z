#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:24:29 2018

@author: standarduser
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

# dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

# fitting hc to dataset
from sklearn.cluster import AgglomerativeClustering
hc =AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0,1], X[y_hc == 0,0],s = 100, c= 'red',label= 'Cluster1')
plt.scatter(X[y_hc == 1,1], X[y_hc == 1,0],s = 100, c= 'blue',label= 'Cluster2')
plt.scatter(X[y_hc == 2,1], X[y_hc == 2,0],s = 100, c= 'green',label= 'Cluster3')
plt.scatter(X[y_hc == 3,1], X[y_hc == 3,0],s = 100, c= 'cyan',label= 'Cluster4')
plt.scatter(X[y_hc == 4,1], X[y_hc == 4,0],s = 100, c= 'magenta',label= 'Cluster5')

plt.legend()
plt.show()

