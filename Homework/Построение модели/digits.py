from sklearn import datasets
from  sklearn.linear_model import LogisticRegression
from  sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



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
# print(Y_test)
# print(Y_train)
lgr = LogisticRegression()

lgr.fit(X_train,Y_train)
Y_pred = lgr.predict(X_test)

print(accuracy_score(Y_test, Y_pred))

dtclf = DecisionTreeClassifier()

dtclf.fit(X_train,Y_train)
Y_pred = dtclf.predict(X_test)

print(accuracy_score(Y_test, Y_pred))

knn = KNeighborsClassifier()

knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)

print(accuracy_score(Y_test, Y_pred))
