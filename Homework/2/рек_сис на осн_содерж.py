import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split

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
# print()

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

def change_string(s):
    return ' '.join(s.replace(' ', '').replace('-', '').split('|'))
#
movie_genres = [change_string(g) for g in data.genres.values]
# print(data.info())
tag_strings = []
movies = []
# movie_genres = []

# for movie, group in data.groupby('title'):
#     movie_genres.append(' '.join([str(g).replace(' ', '').replace('-', '').split('|') for g in group.genres.values]))
#     movies.append(movie)

# def gen():
#     for k in data.genres.values:
#         movie_genres.append(' '.join(str(k).replace(' ', '').replace('-', '').split('|')))
for movie, group in data.groupby('title'):
    # gen()
    # movie_genres.append(g.replace(' ', '').replace('-', '').split('|') for g in group.genres.values)
    tag_strings.append(' '.join([str(s).replace(' ', '').replace('-', '') for s in group.tag.values]))
    movies.append(movie)
# print(len(movies))
# print(len(movie_genres))
# print(len(tag_strings))
movies_for_tags_df = pd.DataFrame({'title':movies})
movies_for_genres_df = data['title']
# print(movies.sha)
# print()
# t = pd.Series(tag_strings)
genres_df = pd.DataFrame({'genres':movie_genres})
tag_strings_df =pd.DataFrame({'tag':tag_strings})

movies_tags_df =pd.concat([movies_for_tags_df,tag_strings_df])
movies_genres_df = pd.concat([movies_for_genres_df,genres_df])
# print(movies_tags_df.info())
# print(movies_tags_df.head())
# print(movies_genres_df.head())
# print(movies_genres_df.describe())
# print(movies_tags_df.describe())
# print(tag_strings_df.describe())
# mov_tag_genre = pd.merge()


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
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
x = pd.merge(genres_full,tags_full,how='left')
print(x.head())
# print(tf_genres.shape)
# print(tf_genres_df.shape)
#
# print(tf_tag.shape)
# print(tf_tag_df.shape)
#
# print(tags_full.shape)
# print(genres_full.shape)
#
# print(tags_full.head())
# print(genres_full.head())
