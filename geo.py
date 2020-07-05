import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np



data = pd.read_excel('geo.xlsx', index_col=0)

x = data.drop('comment_class',axis='columns')
y = data['comment_class']


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

clf = KMeans()
clusters =clf.fit_predict(x_scaled, )


plt.scatter(x_scaled[:,0],x_scaled[:,1],c=clusters, cmap='autumn', s=60,label ='Без голосов')
plt.legend()
plt.show()


clf = KMeans()
clusters =clf.fit_predict(x_scaled, )

plt.scatter(x_scaled[:,0],x_scaled[:,1],c=y, cmap='autumn', s=60, label='С голосами')
plt.legend()
plt.show()


k_inertia = []
ks = range(1,11)

for k in ks:
    clf_kmeans = KMeans(n_clusters=k)
    clusters_kmeans = clf_kmeans.fit_predict(x_scaled, )
    k_inertia.append(clf_kmeans.inertia_)

plt.plot(ks, k_inertia)
plt.plot(ks, k_inertia ,'ro')
plt.show()


diff = np.diff(k_inertia)

plt.plot(ks[1:], diff)
plt.show()


diff_r = diff[1:] / diff[:-1]

plt.plot(ks[1:-1], diff_r)
plt.show()


k_opt = ks[np.argmin(diff_r)+1]
print(k_opt)


# fig, axes = plt.subplots(1, 5, figsize=(15, 2))
# alpha = 0.3
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_random, cmap='autumn', s=60, alpha=alpha)
# axes[0].set_title(get_descr("Random", y, clusters_random, X_scaled))
# for ax, algorithm in zip(axes[1:], algorithms):
#     # кластеризуем и выводим картинку
#     clusters = algorithm.fit_predict(X_scaled)
#     ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='autumn', s=60, alpha=alpha)
#     ax.set_title(get_descr(algorithm.__class__.__name__, y, clusters, X_scaled))

#     # если есть центры кластеров - выведем их
#     if algorithm.__class__.__name__ in {'KMeans', 'AffinityPropagation'}:
#         centers = algorithm.cluster_centers_
#         ax.scatter(centers[:, 0], centers[:, 1], s=50)