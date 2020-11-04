import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


train = np.loadtxt(r'C:\Users\pc\PycharmProjects\Netology\Classwork\CV\002\data\digit\train.csv', delimiter=',', skiprows=1)
# test = np.loadtxt(r'C:\Users\pc\PycharmProjects\Netology\Classwork\CV\002\data\digit\train.csv', delimiter=',', skiprows=Построение модели)

# сохраняем разметку в отдельную переменную
train_label = train[:, 0]


# приводим размерность к удобному для обаботки виду
train_img = np.reshape(train[:, 1:], (len(train[:, 1:]), 28, 28))

# выбираем семпл данных для обработки
choices = np.random.choice(train_img.shape[0], 10000)

y = train_label[choices]
X = train_img[choices].reshape(-1, 28 * 28).astype(np.float32)

# центрируем данные
X_mean = X.mean(axis=0)
X -= X_mean
print(X.shape)

cumsum_list = []
number_of_components = list(range(25,200))
def pca(comp):
    pca = PCA(n_components=comp)
    pca.fit(X)
    S = pca.explained_variance_ratio_
    S = sum(S)
    V = pca.components_
    return V

v = pca(784)

Xrot_reduced = np.dot(X, v[:, :150])
print(Xrot_reduced.shape,y.shape)

x_train, x_test, y_train, y_test = train_test_split(Xrot_reduced,y,test_size=0.3,random_state=1)

lr = LogisticRegression(random_state=1,n_jobs=-1,penalty='l2')
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

print('accuracy',accuracy_score(y_test,y_pred))
# accuracy 0,844