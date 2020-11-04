from  sklearn.linear_model import LogisticRegression
from  sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt



titanic = pd.read_csv('../data/titanic.csv')
# FE


le = preprocessing.LabelEncoder()
titanic['Sex'] = le.fit_transform(titanic['Sex'])

titanic = titanic.drop(['Name'], axis='columns')
X = titanic.drop(['Survived'], axis=1)
y = titanic['Survived']
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3)

lgr = LogisticRegression()

lgr.fit(X_train,y_train)
Y_pred_proba = lgr.predict_proba(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pred_proba[: ,1])
plt.plot(fpr,tpr,label="data Построение модели")
plt.legend(loc=4)
plt.show()


dtclf = DecisionTreeClassifier()

dtclf.fit(X_train,y_train)
Y_pred_proba = dtclf.predict_proba(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pred_proba[: ,1])
plt.plot(fpr,tpr,label="data Построение модели")
plt.legend(loc=4)
plt.show()

y_pred = dtclf.predict(X_test)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)
Y_pred_proba = knn.predict_proba(X_test)



fpr, tpr, _ = metrics.roc_curve(y_test,  Y_pred_proba[: ,1])
plt.plot(fpr,tpr,label="data Построение модели")
plt.legend(loc=4)
plt.show()




