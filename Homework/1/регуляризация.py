import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV




desired_width=320
pd.set_option('display.width', desired_width)
# np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',30)

data = pd.read_csv(r'/Homework/data/adult.csv')
y = data['income']
x = data.drop('income',axis=1)
x = pd.get_dummies(x)
lbe = LabelEncoder()
y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)



params = [ {'max_depth': list( range(1, 20) )} ]
gs = GridSearchCV( DecisionTreeClassifier(), param_grid = params,)
gs.fit(x_train,y_train)
params = gs.best_params_
val = list((params.values()))[0]
model = DecisionTreeClassifier(max_depth=val)
model.fit(x_train,y_train)
y_pred = model.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test,y_pred[:,1])
plt.plot( fpr, tpr,)
plt.title(label ='DesTreeCLF')
plt.legend()
plt.show()




for i in range(1,5):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred = knn.predict_proba(x_test)

    fpr, tpr, thresholds = roc_curve(y_test,y_pred[:,1])
    plt.plot( fpr, tpr,label = 'KNN neighbors '+ str(i)  )
    plt.legend()
    plt.show()





LR = LogisticRegression(penalty='l2', C = 0.1)
LR.fit(x_train,y_train)
y_pred = LR.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test,y_pred[:,1])
plt.plot( fpr, tpr, label= 'LR l2 c=0.1' )
plt.legend()
plt.show()



LR = LogisticRegression(penalty='l2', C = 0.01)
LR.fit(x_train,y_train)
y_pred = LR.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test,y_pred[:,1])
plt.plot( fpr, tpr,  label= 'LR l2 c=0.01' )
plt.legend()
plt.show()

