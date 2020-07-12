import pandas as pd


# Настройка вывода таблиц окна run
desired_width=320
pd.set_option('display.width', desired_width)
# np.set_printoption(linewidth=desired_width)
pd.set_option('display.max_columns',30)

data = pd.read_csv(r'homework/data\train.csv')

# Просмотр типов и количества значений в сете
print(data.info())

# Вывод верхушки NaN значений в столюце 'height'
print(data[ pd.isnull( data['height'] ) ].head())

# Вывод количества пропущенных значений в стобце 'height'
print('Для height пустых строк ' + str( len( data[ pd.isnull( data['height'] ) ] ) ))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder()

ft = ohe.fit_transform(data[['nationality', 'height', 'weight',
 'sport', 'gold', 'silver', 'bronze']])


Xshort = pd.get_dummies( data.loc[ :, ['age', 'children'] ],
 columns = ['age', 'children'] )