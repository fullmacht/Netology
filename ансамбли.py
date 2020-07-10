import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


df_train = pd.read_csv(r'C:\Users\laptop\PycharmProjects\des_tree_clf_titanik\data\train.csv')
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

df_train = pd.get_dummies(df_train)

x = df_train.drop(['SalePrice','Id'],axis=1)
y = df_train['SalePrice']

model = RandomForestClassifier()
model.fit(x,y)

imp = pd.Series(model.feature_importances_)
print(imp.sort_values(ascending=False))


classifier = StackingClassifier(
    [   ('rf',RandomForestClassifier()),
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier())
    ],
LogisticRegression())

classifier.fit(x,y)

plt.plot(classifier.final_estimator_.coef_.flatten(),classifier.named_estimators_.keys())
plt.show()


