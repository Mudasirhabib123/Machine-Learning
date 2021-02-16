import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

features, _ = make_blobs(n_samples = 50, n_features = 2, centers = 3, random_state = 1)

df = pd.DataFrame(features, columns=['feature 1', 'feature 2'])
cluster = KMeans(3,random_state = 0)
cluster.fit(features)

df['Group'] = cluster.predict(features)

print(df.tail())