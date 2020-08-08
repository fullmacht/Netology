import pandas as pd
from surprise import KNNWithMeans,KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

ratings =pd.read_csv(r'C:\Users\laptop\PycharmProjects\Netology\Classwork\data\ratings.csv')
movies =pd.read_csv(r'C:\Users\laptop\PycharmProjects\Netology\Classwork\data\movies.csv')


movies_with_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)

dataset = pd.DataFrame({
    'uid': movies_with_ratings.userId,
    'iid': movies_with_ratings.title,
    'rating': movies_with_ratings.rating
})

print(ratings.rating.min())
print(ratings.rating.max())

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(dataset, reader)

trainset, testset = train_test_split(data, test_size=.15,random_state=1)


for i in range(1,101):
    algo = KNNWithMeans(k=i, sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo.fit(trainset)
    test_pred = algo.test(testset)
    print(i,'-------------',accuracy.rmse(test_pred, verbose=True))
#8861 -61
