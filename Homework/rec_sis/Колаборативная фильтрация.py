import pandas as pd
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
import operator

ratings =pd.read_csv(r'C:\Users\laptop\PycharmProjects\Netology\Classwork\data\ratings.csv')
movies =pd.read_csv(r'C:\Users\laptop\PycharmProjects\Netology\Classwork\data\movies.csv')


movies_with_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)

dataset = pd.DataFrame({
    'uid': movies_with_ratings.userId,
    'iid': movies_with_ratings.title,
    'rating': movies_with_ratings.rating
})

min_r = ratings.rating.min()
max_r = ratings.rating.max()

reader = Reader(rating_scale=(min_r, max_r))
data = Dataset.load_from_df(dataset, reader)

trainset, testset = train_test_split(data, test_size=.15,random_state=1)

d = {}
for i in range(40,42):
    algo = KNNWithMeans(k=i, sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo.fit(trainset)
    test_pred = algo.test(testset)
    a = accuracy.rmse(test_pred, verbose=False)
    d[i] = a
    print('Завершено предсказание при k = {}'.format(i))
mi = min(d.items(), key=operator.itemgetter(1))[0]
print('Минимальная rmse = {} при k = {}'.format(d[mi],mi))

#user_b Минимальный rmse = 0.88 при k = 61
#item_b  Минимальная rmse = 0.8689299715492731 при k = 41
