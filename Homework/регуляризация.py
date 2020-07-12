import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve


data = pd.read_csv(r'/data/adult.csv')
y = data['income']
x = data.drop('income',axis=1)
x = pd.get_dummies(x)
# y = pd.get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# y_train = y_train['<=50K'].append(y_train['>50K']).reset_index(drop=True)
# y_test = y_test['<=50K'].append(y_test['>50K']).reset_index(drop=True)
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)
y_pred = dt_clf.predict_proba(x_test)
print(roc_curve(y_test,y_pred[:,1]))