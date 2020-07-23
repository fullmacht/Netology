from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


X, y = load_boston(return_X_y=True)

k_range = list(range(1, 31))
grid_param = dict(n_neighbors=k_range,leaf_size=k_range)
gs = GridSearchCV(KNeighborsRegressor(),param_grid=grid_param,scoring='neg_mean_squared_error',n_jobs=-1,cv=10)
gs.fit(X,y)
print(gs.best_score_)



grid_param = {
    'criterion' : ['mse', 'friedman_mse', 'mae'],
    'max_depth' : k_range
}
gs = GridSearchCV(DecisionTreeRegressor(),grid_param,scoring='neg_mean_squared_error',n_jobs=-1, cv=5)
gs.fit(X,y)
print(gs.best_score_)



grid_param = {
    'normalize':[True,False]
}
gs = GridSearchCV(LinearRegression(), grid_param, scoring='neg_mean_squared_error',n_jobs=-1, cv=5)
gs.fit(X,y)
print(gs.best_score_)
