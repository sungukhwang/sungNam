# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:07:28 2017

@author: pc
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

ds = pd.read_csv("C:\\Users\\pc\\Desktop\\c.csv")

#(1) c.csv 화일을 읽어 data set (이름 : ds)을 만드시오.

ds = ds.dropna(axis=0)

X = ds[["X1","X2","X3","X4","X5","X6","X7"]]
y = ds[["Y"]]

import mglearn

mglearn.plots.plot_cross_validation()

# 데이터를 훈련+검증 세트 그리고 테스트 세트로 분할
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, random_state=0)
# 훈련+검증 세트를 훈련 세트와 검증 세트로 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)
print("훈련 세트의 크기: {}   검증 세트의 크기: {}   테스트 세트의 크기:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

from sklearn import grid_search, cross_validation

grid_search.GridSearchCV()

gs = grid_search.GridSearchCV()
estimator=LogisticRegression(),
param_grid={'C': [10**-i for i in range(-5, 5)], 'class_weight': [None, 'auto']},
cv=cross_validation.KFold(n=len(ds), n_folds=10)

gs.fit(d_features, titanic.Y1)

gs.best_estimator_

