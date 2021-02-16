from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


features, _ = make_blobs(n_features=5, n_samples=10,random_state=0)

model = KMeans(n_clusters=2,random_state=None)
model.fit(features)

predicted = model.labels_

silh = silhouette_score(features, predicted)
print(silh)
print('features', features)
print('\npredicted', predicted)
print('\nacualcted', _)
