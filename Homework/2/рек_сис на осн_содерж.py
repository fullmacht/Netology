import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',30)


links = pd.read_csv(r'C:\Users\laptop\PycharmProjects\des_tree_clf_titanik\Classwork\data\links.csv')
movies = pd.read_csv(r'C:\Users\laptop\PycharmProjects\des_tree_clf_titanik\Classwork\data\movies.csv')
ratings = pd.read_csv(r'C:\Users\laptop\PycharmProjects\des_tree_clf_titanik\Classwork\data\ratings.csv')
tags = pd.read_csv(r'C:\Users\laptop\PycharmProjects\des_tree_clf_titanik\Classwork\data\tags.csv')


ratings = ratings.drop(['timestamp','userId'],axis=1)
data = pd.merge(movies,tags, left_on='movieId',right_on='movieId')
data = pd.merge(data,ratings, left_on='movieId',right_on='movieId')
data = data.drop(['userId','timestamp'],axis=1)
data.dropna(inplace=True)


med = data.groupby(['movieId']).rating.mean()
data = pd.merge(data,med, left_on='movieId',right_on='movieId')
data = data.drop('rating_x',axis=1)
y = data['rating_y']
x = data.drop('rating_y',axis=1)
x = x.groupby(['movieId'])


def change_string(s):
    return ' '.join(s.replace(' ', '').replace('-', '').split('|'))

movie_genres = [change_string(g) for g in data.genres.values]


tag_strings = []
movies = []

for movie, group in data.groupby('title'):
    tag_strings.append(' '.join([str(s).replace(' ', '').replace('-', '') for s in group.tag.values]))
    movies.append(movie)


movies_for_tags_df = pd.DataFrame({'title':movies})
movies_for_genres_df = data['title']

genres_df = pd.DataFrame({'genres':movie_genres})
tag_strings_df =pd.DataFrame({'tag':tag_strings})

movies_tags_df =pd.concat([movies_for_tags_df,tag_strings_df])
movies_genres_df = pd.concat([movies_for_genres_df,genres_df])


cv = CountVectorizer()
genres_vec = cv.fit_transform(movie_genres)
tag_vec = cv.fit_transform(tag_strings)
tf = TfidfTransformer()
tf_genres = tf.fit_transform(genres_vec)
tf_tag = tf.fit_transform(tag_vec)
tf_genres_df = pd.DataFrame.sparse.from_spmatrix(tf_genres)
tf_tag_df = pd.DataFrame.sparse.from_spmatrix(tf_tag)
genres_full = pd.concat([movies_for_genres_df,tf_genres_df],axis=1)
tags_full = pd.concat([movies_for_tags_df,tf_tag_df],axis=1)
x = pd.merge(genres_full,tags_full,on='title')

x = x.fillna(0)
y = y.fillna(0)


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)
y_pred = dtr.predict(x_test)

rmse = mean_squared_error(y_test,y_pred,squared=False)
print(rmse)