import numpy as np 
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


digits = load_digits()
features = StandardScaler().fit_transform(digits.data)
pca = PCA(n_components= .99,whiten= True)

features_pca = pca.fit_transform(features)

print('origional features', features.shape[1])
print('pca features', features_pca.shape[1])
