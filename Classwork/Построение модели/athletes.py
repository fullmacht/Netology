import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score



data = pd.read_csv(r'C:\Users\pc\PycharmProjects\Homework\data\athletes.csv')
data = data[ pd.isnull( data['height'] ) == 0 ]
data = data[ pd.isnull( data['weight'] ) == 0 ]
pd.get_dummies(data,columns=['name',])
print(data.info())
x = data.drop('sex',axis=1)
y = data['sex']
# print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

model = LogisticRegression()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)

precis_sc = precision_score(y_test,y_pred)
print(precis_sc)
