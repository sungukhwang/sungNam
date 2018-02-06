# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:17:45 2017

@author: pc
"""
import numpy as np
import pandas as pd

train = np.array("C:\\Users\\pc\\Desktop\\train.csv")

train.dropna()
train = train.dropna(axis=0)

mean_fare = train["Fare"].mean()
print("Fare(Mean) = ${0:.3f}".format(mean_fare))

from sklearn.feature_selection import SelectPercentile, f_classif

# 고정된 난수를 발생시킵니다
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(train), 50))
# 데이터에 노이즈 특성을 추가합니다
# 처음 30개는 원본 특성이고 다음 50개는 노이즈입니다
X_w_noise = np.hstack([train, noise])

rng = np.random.RandomState(42)
noise = rng.normal(size=(len(test), 50))
# 데이터에 노이즈 특성을 추가합니다
# 처음 30개는 원본 특성이고 다음 50개는 노이즈입니다
X_w_noise = np.hstack([test, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, train, random_state=0, test_size=.5)
# f_classif(기본값)와 SelectPercentile을 사용하여 특성의 50%를 선택합니다
select = SelectPercentile(score_func=f_classif, percentile=50)
select.fit(X_train, y_train)
# 훈련 세트에 적용합니다
X_train_selected = select.transform(X_train)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

