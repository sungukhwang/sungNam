# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:38:05 2017

@author: pc
"""
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

#################################################################

ds = pd.read_csv("C:\\Users\\pc\\Desktop\\c.csv")

#(1) c.csv 화일을 읽어 data set (이름 : ds)을 만드시오.

ds = ds.dropna(axis=0)

print("원본 특성:\n", list(ds.columns), "\n")
ds = pd.get_dummies(ds)

print("get_dummies 후의 특성:\n", list(ds.columns))

display(ds.head())
#종속변수와 독립변수 분리
features = ds.loc[:, 'X1':'X7']
# NumPy 배열 추출
X = features.values
y = ds['Y'].values
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

####
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

####
from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print("예측된 레이블의 고유값: {}".format(np.unique(pred_most_frequent)))
print("테스트 점수: {:.2f}".format(dummy_majority.score(X_test, y_test)))

####
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("테스트 점수: {:.2f}".format(tree.score(X_test, y_test)))

####
from sklearn.linear_model import LogisticRegression

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("dummy 점수: {:.2f}".format(dummy.score(X_test, y_test)))

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg 점수: {:.2f}".format(logreg.score(X_test, y_test)))

####
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred_logreg)
print("오차 행렬:\n{}".format(confusion))
### confusion matrix
print("빈도 기반 더미 모델:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\n무작위 더미 모델:")
print(confusion_matrix(y_test, pred_dummy))
print("\n결정 트리:")
print(confusion_matrix(y_test, pred_tree))
print("\n로지스틱 회귀")
print(confusion_matrix(y_test, pred_logreg))

from sklearn.metrics import f1_score
print("빈도 기반 더미 모델의 f1 score: {:.2f}".format(
    f1_score(y_test, pred_most_frequent)))
print("무작위 더미 모델의 f1 score: {:.2f}".format(f1_score(y_test, pred_dummy)))
print("트리 모델의 f1 score: {:.2f}".format(f1_score(y_test, pred_tree)))
print("로지스틱 회귀 모델의 f1 score: {:.2f}".format(
    f1_score(y_test, pred_logreg)))

from sklearn.metrics import classification_report
print(classification_report(y_test, pred_logreg,
                            target_names=["1", "0"]))

######## 스케일 조정
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)#X_train
X_scaled = scaler.transform(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

####

dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train_scaled, y_train)
pred_most_frequent = dummy_majority.predict(X_test_scaled)
print("조정 후 예측된 레이블의 고유값: {}".format(np.unique(pred_most_frequent)))
print("조정 후 테스트 점수: {:.2f}".format(dummy_majority.score(X_test_scaled, y_test)))

####
tree = DecisionTreeClassifier(max_depth=2).fit(X_train_scaled, y_train)
pred_tree = tree.predict(X_test_scaled)
print("조정 후 테스트 점수: {:.2f}".format(tree.score(X_test_scaled, y_test)))

####

dummy = DummyClassifier().fit(X_train_scaled, y_train)
pred_dummy = dummy.predict(X_test_scaled)
print("조정 후 dummy 점수: {:.2f}".format(dummy.score(X_test_scaled, y_test)))

logreg = LogisticRegression(C=0.1).fit(X_train_scaled, y_train)
pred_logreg = logreg.predict(X_test_scaled)
print("조정 후 logreg 점수: {:.2f}".format(logreg.score(X_test_scaled, y_test)))

print("오차 행렬:\n{}".format(confusion))
### confusion matrix
print("조정 후 빈도 기반 더미 모델:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\n조정 후 무작위 더미 모델:")
print(confusion_matrix(y_test, pred_dummy))
print("\n조정 후 결정 트리:")
print(confusion_matrix(y_test, pred_tree))
print("\n조정 후 로지스틱 회귀")
print(confusion_matrix(y_test, pred_logreg))

print("조정 후 빈도 기반 더미 모델의 f1 score: {:.2f}".format(
    f1_score(y_test, pred_most_frequent)))
print("조정 후 무작위 더미 모델의 f1 score: {:.2f}".format(f1_score(y_test, pred_dummy)))
print("조정 후 트리 모델의 f1 score: {:.2f}".format(f1_score(y_test, pred_tree)))
print("조정 후 로지스틱 회귀 모델의 f1 score: {:.2f}".format(
    f1_score(y_test, pred_logreg)))

from sklearn.metrics import classification_report
print(classification_report(y_test, pred_logreg,
                            target_names=["1", "0"]))

#####################svm의 그래프
svc = SVC(gamma=.05).fit(X_train_scaled, y_train)

precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test_scaled))
# 0에 가까운 임계값을 찾습니다
close_zero = np.argmin(np.abs(thresholds))

################랜덤포레스트와 svc의 그래프

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train_scaled, y_train)

# RandomForestClassifier는 decision_function 대신 predict_proba를 제공합니다.
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
    y_test, rf.predict_proba(X_test_scaled)[:, 1])

plt.plot(precision, recall, label="svc")

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=20,
         label="svc: critical 0", fillstyle="none", c='k', mew=2)

plt.plot(precision_rf, recall_rf, label="rf")

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k',
         markersize=10, label="rf: critical 0.5", fillstyle="none", mew=2)
plt.xlabel("precision")
plt.ylabel("recall")
plt.legend(loc="best")

