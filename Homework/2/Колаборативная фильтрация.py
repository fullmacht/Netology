import pandas as pd
from surprise import KNNWithMeans,KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split

ratings =pd.read_csv(r'C:\Users\laptop\PycharmProjects\Netology\Classwork\data\big_movielens\ratings.csv')
movies =pd.read_csv(r'C:\Users\laptop\PycharmProjects\Netology\Classwork\data\big_movielens\movies.csv')


movies_with_ratings = movies.join(ratings.set_index('movieId'), on='movieId').reset_index(drop=True)
movies_with_ratings.dropna(inplace=True)

dataset = pd.DataFrame({
    'uid': movies_with_ratings.userId,
    'iid': movies_with_ratings.title,
    'rating': movies_with_ratings.rating
})

print(ratings.rating.min())
print(ratings.rating.max())