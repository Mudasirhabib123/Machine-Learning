from sklearn.datasets import load_digits
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import numpy as np 

digits = load_digits()
features = digits.data

features = StandardScaler().fit_transform(features)

sparse_features = csr_matrix(features)


tsvd = TruncatedSVD(n_components=10)

tsvd_features = tsvd.fit_transform(sparse_features)

print(features.shape)
print(sparse_features.shape)
print(tsvd_features.shape)
