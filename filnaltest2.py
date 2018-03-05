# -*- coding: utf-8 -*-
# -*- coding: cp949 -*-
"""
Created on Sun Dec 17 14:37:50 2017

@author: pc
"""
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas import DataFrame
plt.rc('font', family='Malgun Gothic')

ds = pd.read_csv("C:\\Users\\pc\\Desktop\\c2.csv")
'''1. 종속변수로 Y1, 독립변수로 X1, X2, X3, X5, X7을 이용하여 
  (1) Y1의 값을 P는 1로, F는 0으로 바꾸시오.
  (2) data set를 training, validation, test set으로 나누시오.
  (3) Logistic regression, SVM, Decision Tree 알고리즘을 이용하여 분류분석을 수행하고,
  그 결과를 평가지표를 2개 이상 사용하여 비교 설명하시오.
 '''
# X4,X6,Y2 변수들을 제거.
#(1)
del ds["X4"]
del ds["X6"]
del ds["Y2"]
#(2)
ds.loc[ds["Y1"] == "P", "Y1"] = '1'
ds.loc[ds["Y1"] == "F", "Y1"] = '0'

features = ds.loc[:, 'X1':'X7']
X = features.values
y = ds['Y1'].values
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, random_state=0)
# 훈련+검증 세트를 훈련 세트와 검증 세트로 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)
print("\n훈련 세트의 크기: {}   검증 세트의 크기: {}   테스트 세트의 크기:"
      " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

####함수선언
def gegege(model):
    print("훈련 세트 정확도: {:.3f}"  .format(model.score(X_train, y_train)))
    print("테스트 세트 정확도: {:.3f}"  .format(model.score(X_test, y_test)))
##################### (3)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

gegege(logreg)

from sklearn.svm import LinearSVC

linear_svm = LinearSVC()
linear_svm.fit(X_train, y_train)
gegege(linear_svm)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
gegege(tree)

##################
#평가
from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print("예측된 레이블의 고유값: {}".format(np.unique(pred_most_frequent)))
print("DummyClassifier 테스트 점수: {:.2f}".format(dummy_majority.score(X_test, y_test)))

####
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("DecisionTreeClassifier 테스트 점수: {:.2f}".format(tree.score(X_test, y_test)))

####
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg 점수: {:.2f}".format(logreg.score(X_test, y_test)))

####
#로지스틱 오차행렬
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred_logreg)
print("오차 행렬:\n{}".format(confusion))
#tree 오차행렬
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred_tree)
print("오차 행렬:\n{}".format(confusion))

### confusion matrix

print("\n결정 트리:")
print(confusion_matrix(y_test, pred_tree))

from sklearn.metrics import classification_report
print(classification_report(y_test, pred_logreg,
                            target_names=["1", "0"]))
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, pred_logreg)

print(classification_report(y_test, pred_tree,
                            target_names=["1", "0"]))
confusion = confusion_matrix(y_test, pred_tree)

#################


'''
결론 : c2 data set은 서포트벡터머신으로 분류한 정확도는 아주 낮았다. 
전체적으로 우수한 정확도를 가지는 것은 Decision Tree 알고리즘이다.
훈련 세트 정확도: 0.847
테스트 세트 정확도: 0.726
또한,

오차 행렬:
[[111  12]
 [ 41  11]]
오차 행렬:
[[105  18]
 [ 34  18]]
결정 트리:
[[105  18]
 [ 34  18]]

             precision    recall  f1-score   support

          1       0.73      0.90      0.81       123
          0       0.48      0.21      0.29        52

avg / total       0.66      0.70      0.65       175

             precision    recall  f1-score   support

          1       0.76      0.85      0.80       123
          0       0.50      0.35      0.41        52

avg / total       0.68      0.70      0.68       175

이러한 판정 결과에 따라 tree의 f1-score와 precision 점수가 더 높은 것을 알 수 있다.

추가 ) 독립변수들 중 편차가 큰 값들이 있어서 SVM의 값이 높지 않은 것.
그렇다면 MinMaxScaler등을 사용해서 스케일 조정을 한다면 더 높은 값을 가질 수 있다.


'''


print("\n")
###########2. 종속변수로 Y2, 독립변수로 X3~X7을 이용하여 
### (1) Random Forest 3겹 교차검증을 실시하고 분류 정확도를 제시하시오.

ds = pd.read_csv("C:\\Users\\pc\\Desktop\\c2.csv")

features = ds.loc[:, 'X3':'X7']
X = features.values
y = ds['Y2'].values
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2)

# 3겹 교차검증
# 데이터를 섞어서 3-겹
from sklearn.model_selection import cross_val_score

# 3겹 교차검증
scores = cross_val_score(forest, X, y)
print("3겹 교차 검증 점수: {}".format(scores))

# 데이터를 섞어서 3-겹

kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print("SHUFFLE 교차 검증 점수:\n{}".format(
    cross_val_score(forest, X, y, cv=kfold)))
print("SHUFFLE교차 검증 평균 점수: {:.2f}".format(scores.mean()))

'''
결론: 랜덤 포레스트를 이용한 그냥 3겹 교차검증보다 데이터를 섞는 3겹 교차 검증이 더 좋았다.
그 이유는 당연히 Y2변수에 있는 값들이 특이한 형태를 취하고 있어서 이다. 
이런 특이한 데이터를 판별해야 한다.
3겹 교차 검증 점수: [ 0.45726496  0.45922747  0.48927039]
SHUFFLE 교차 검증 점수:
[ 0.48290598  0.472103    0.51502146]
SHUFFLE교차 검증 평균 점수: 0.47
'''

'''
3. 독립변수로 X3~X7을 이용하여 
  (1) X1의 값을 이용하여 2개의 범주를 갖는 새로운 종속변수 Y3를 만드시오
  (X1이 40보다 작으면 Y3=0, 그렇지 않으면 Y3=1).
  (2) data set를 training, test set으로 나누시오.
  (3) Decision Tree, Random Forest 알고리즘을 이용하여 분류분석을 수행하고, 
  각 알고리즘의 f-score와 AUC를 비교 설명하시오.
  '''

##################(1)
ds = pd.read_csv("C:\\Users\\pc\\Desktop\\c2.csv")

ds["Y3"] = 0

for i in range(X.shape[0]):
    if (int(ds.ix[i,"X1"])<40):
        ds.ix[i,'Y3']=0
    else :
        ds.ix[i,'Y3']=1
features = ds.loc[:, 'X3':'X7']

X = features.values
y = ds['Y3'].values

###################(2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

###################(3)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2)


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

### f1 score
from sklearn.metrics import f1_score
print("트리 모델의 f1 score: {:.2f}".format(f1_score(y_test, pred_tree)))
print("랜덤포레스트 회귀 모델의 f1 score: {:.2f}".format(f1_score(y_test, pred_forest)))

from sklearn.metrics import roc_auc_score
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
tree_auc = roc_auc_score(y_test, tree.decision_function(X_test))
print("랜덤 포레스트의 AUC: {:.3f}".format(rf_auc))
print("SVC의 AUC: {:.3f}".format(tree_auc))

