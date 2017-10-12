#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:11:48 2017

@author: dpekach
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from scipy import stats
import math
from scipy.spatial import distance
import pdb

 
iris = datasets.load_iris()

X = iris.data
y = np.expand_dims(iris.target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
clf = KNeighborsClassifier(n_neighbors = 10)
clf.fit(X_train, y_train)
print(accuracy_score(y_test,clf.predict(X_test)))

def classifyKNN (X_train, y_train, X_test, k):
    def calculate_distance(test_example, train_example):
        return distance.euclidean(test_example, train_example)
    distance_list = []
    label_list = [] 
    for test_example in X_test:
        for i in range(len(X_train)):
            element = (calculate_distance(test_example, X_train[i]), y_train[i].tolist())
            distance_list.append(element)
        distance_list = sorted(distance_list, key = lambda x: x[0])
        mode_list = [i[1][0] for i in distance_list[0:k]]
        label_list.append(stats.mode(mode_list)[0].tolist()[0])
        distance_list = []
    return list(label_list)

y_pred = classifyKNN(X_train, y_train, X_test, 10)
print(accuracy_score(y_test, y_pred))

    