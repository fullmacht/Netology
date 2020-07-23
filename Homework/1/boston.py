from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
X, y = load_boston(return_X_y=True)
y = pd.DataFrame(y)
X = pd.DataFrame(X)
print(y.shape)
print(X.shape)
# print(np.median(y))
le = LabelEncoder()
y_pre = le.fit_transform(y)
# ohe = OneHotEncoder()
# y_pre = ohe.fit_transform(y)
print(y_pre)
grid_param = {
    'n_neighbors':range(1,26),
    'leaf_size' : range(2,30,2)
}
gs = GridSearchCV(KNeighborsClassifier(),grid_param,scoring='roc_auc',n_jobs=-1,cv=5)
gs.fit(X,y_pre)
print(gs.best_score_)
for k in gs.cv_results_:
    print(k, ":", gs.cv_results_[k][0])
grid_param = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(1,31)
}
gs = GridSearchCV(DecisionTreeClassifier(),grid_param,scoring='accuracy',n_jobs=-1, cv=6)
gs.fit(X,y_pre)
print(gs.best_score_)