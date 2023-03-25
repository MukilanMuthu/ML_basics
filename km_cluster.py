import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)  # data created will be a
#                                                                                               tuple 0 will be the data
#                                                                                               1 will be the
#                                                                                               class it belongs to
# plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')
kmeans = KMeans(n_clusters=4)  # since we already know the number of clusters
kmeans.fit(data[0])
label = kmeans.labels_

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
ax1.set_title('K Means')
ax1.scatter(data[0][:,0], data[0][:,1], c=label, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')

plt.show()
