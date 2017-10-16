#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:11:48 2017

@author: dpekach
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from scipy import stats
import math
from scipy.spatial import distance
import pdb

from nearpy import Engine
import nearpy
from nearpy.filters import NearestFilter, UniqueFilter
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import CosineDistance

import time

import warnings
warnings.filterwarnings("ignore")

iris = datasets.load_iris()

X = iris.data
y = np.expand_dims(iris.target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                        y, test_size = 0.2, random_state = 0)

def classifyKNN (X_train, y_train, X_test, k):
    def calculate_distance(test_example, train_example):
        return distance.euclidean(test_example, train_example)
    distance_list = []
    label_list = [] 
    for test_example in X_test:
        for i in range(len(X_train)):
            element = (calculate_distance(test_example, X_train[i]), 
                       y_train[i].tolist())
            distance_list.append(element)
        distance_list = sorted(distance_list, key = lambda x: x[0])
        mode_list = [i[1][0] for i in distance_list[0:k]]
        label_list.append(stats.mode(mode_list)[0].tolist()[0])
        distance_list = []
    return list(label_list)
start_time = time.time()
y_pred = classifyKNN(X_train, y_train, X_test, 5)
diff_time = time.time() - start_time
print('Time difference: knn', diff_time)
print('Accuracy score: K Nearest Neighbors', accuracy_score(y_test, y_pred))


def test_nearpy(X_train, y_train, X_test, k):
    # We are looking for the k closest neighbours
    nearest = NearestFilter(k)
    X_train_normalized = []
    for i in range(len(X_train)):
        train_example = X_train[i]
        element = ((train_example / np.linalg.norm(train_example)).tolist(), 
                   y_train[i].tolist())
        X_train_normalized.append(element)
        
    engine = Engine(X_train.shape[1], 
                    lshashes=[RandomBinaryProjections('default', 10)],
                    distance = CosineDistance(),
                    vector_filters = [nearest])
    
    #perform hashing for train examples
    for train_example in X_train:
        engine.store_vector(train_example)
    
    labels = []
    for test_example in X_test:
        neighbors = engine.neighbours(test_example)
        labels.append([train_example[1] for train_example in X_train_normalized
                       if set(neighbors[0][0]) == set(train_example[0])])
    return labels
start_time = time.time()
y_pred = [label[0] for label in test_nearpy(X_train, y_train, X_test, 10)]
diff_time = time.time() - start_time
print('Time difference: ann', diff_time)
print('Accuracy score: Approximate Nearest Neighbors with LSH', accuracy_score(y_test, y_pred))
