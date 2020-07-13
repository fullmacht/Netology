from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score



digits = datasets.load_digits()

X_digits = digits.data
Y_digits = digits.target

n_samples = len( X_digits )
# print(n_samples)

# print(int( n_samples * .9 ))
split, split2 = int( n_samples * .9 ), int( n_samples * .1 )
# print(split,split2)
X_train = X_digits[:split]
Y_train = Y_digits[:split]

X_test = X_digits[:-split2]
Y_test = Y_digits[:-split2]

# Y_train_prep =Y_train

depths = np.arange(1,20)
grid = {'max_depth': depths}#, 'max_features': features_num}
gridsearch = GridSearchCV(DecisionTreeClassifier(), grid, scoring='neg_log_loss', cv=5)
gridsearch.fit(X_train, Y_train)
cross_val_score(gridsearch,X_train,Y_train,cv=5,scoring='accuracy')

scores = [-x for x in gridsearch.cv_results_['mean_test_score']]
plt.plot(depths, scores)
plt.scatter(depths, scores)
best_point = np.argmin(scores)
plt.scatter(depths[best_point], scores[best_point], c='g', s=100)
plt.show()