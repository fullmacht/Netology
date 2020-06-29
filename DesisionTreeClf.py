from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


titanic = pd.read_csv('titanic.csv')
# FE


le = preprocessing.LabelEncoder()
titanic['Sex'] = le.fit_transform(titanic['Sex'])

titanic = titanic.drop(['Name'], axis='columns')
X = titanic.drop(['Survived'], axis=1)
y = titanic['Survived']
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3)






#GridSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
from matplotlib import pyplot as plt

depths = np.arange(1,10)
grid = {'max_depth': depths}
gridsearch = GridSearchCV(DecisionTreeClassifier(), grid, scoring='neg_log_loss', cv=5)

gridsearch.fit(X_train, y_train)

scores = [-x for x in gridsearch.cv_results_['mean_test_score']]
plt.plot(depths, scores)
plt.scatter(depths, scores)
best_point = np.argmin(scores)
plt.scatter(depths[best_point], scores[best_point], c='g', s=100)
plt.show()

clf_final = DecisionTreeClassifier(max_depth=2)

clf_final.fit(X_train, y_train)
y_pred_proba = clf_final.predict_proba(X_test)
y_pred = clf_final.predict(X_test)
print('y_pred_proba',y_pred_proba )
print('y_pred', y_pred )





#Make visualisation of Tree
from sklearn.tree import export_graphviz


def get_tree_dot_view(clf, feature_names=None, class_names=None):
    print(export_graphviz(clf, out_file=None, filled=True, feature_names=feature_names, class_names=class_names))


get_tree_dot_view(clf_final, list(X_train.columns), list(le.classes_))


def get_model(estimator, parameters, X_train, y_train, scoring):
    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator_






# Searching of important features
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score


scoring = make_scorer(accuracy_score, greater_is_better=True)

dtclf = DecisionTreeClassifier(max_depth=2)
parameters = {}
clf_rfc1 = get_model(dtclf, parameters, X_train, y_train, scoring)

plt.figure(figsize=(10,6))
plt.barh(np.arange(X_train.columns.shape[0]), clf_rfc1.feature_importances_, 0.5)
plt.yticks(np.arange(X_train.columns.shape[0]), X_train.columns)
plt.grid()
plt.xticks(np.arange(0,0.2,0.02));
plt.show()