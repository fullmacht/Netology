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


