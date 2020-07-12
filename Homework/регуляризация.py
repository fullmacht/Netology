import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt



desired_width=320
pd.set_option('display.width', desired_width)
# np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',30)

data = pd.read_csv(r'C:\Users\laptop\PycharmProjects\des_tree_clf_titanik\Homework\data\adult.csv')
y = data['income']
x = data.drop('income',axis=1)
x = pd.get_dummies(x)
lbe = LabelEncoder()
y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)
y_pred = dt_clf.predict_proba(x_test)
print('DesTreeCLF',roc_curve(y_test,y_pred[:,1]))
fpr, tpr, thresholds = roc_curve(y_test,y_pred[:,1])
plt.plot( fpr, tpr )
plt.show()

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred = knn.predict_proba(x_test)
print('KNN',roc_curve(y_test,y_pred[:,1]))

fpr, tpr, thresholds = roc_curve(y_test,y_pred[:,1])
plt.plot( fpr, tpr )
plt.show()


LR = LogisticRegression()
LR.fit(x_train,y_train)
y_pred = LR.predict_proba(x_test)
print('LR',roc_curve(y_test,y_pred[:,1]))

fpr, tpr, thresholds = roc_curve(y_test,y_pred[:,1])
plt.plot( fpr, tpr )
plt.show()