# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:20:45 2017

@author: default
"""

from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

X = np.array([[9.96347,4.59677],
[11.033,-0.168167],
[11.5416,5.21116],
[8.69289,1.54322],
[8.10623,4.28696],
[8.30989,4.80624],
[11.9303,4.64866],
[9.67285,-0.202832],
[8.3481,5.13416],
[8.67495,4.47573],
[9.17748,5.09283],
[10.2403,2.45544],
[8.68937,1.4871],
[8.9223,-0.639932],
[9.49123,4.33225],
[9.25694,5.13285],
[7.99815,4.85251],
[8.18378,1.29564],
[8.73371,2.49162],
[9.32298,5.09841],
[10.0639,0.990781],
[9.50049,-0.264303],
[8.34469,1.63824],
[9.50169,1.93825],
[9.15072,5.49832],
[11.564,1.33894]])

y = np.array([1,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,1,0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))

print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

plt.scatter(X[:, 0], X[:, 1], c=y)

fig, ax = plt.subplots()

X1 = X[:,0]
X2 = X[:,1]
ax.scatter(X1[y==0],X2[y==0], marker='^')
ax.scatter(X1[y==1],X2[y==1], marker='s')

plt.show()