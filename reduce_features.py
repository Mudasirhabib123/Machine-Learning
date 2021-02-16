from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits,make_circles
from sklearn.decomposition import PCA,KernelPCA
import matplotlib.pyplot as plt


digits = load_digits()

feature = StandardScaler().fit_transform(digits.data)
# print(digits.data.shape,'\n\n')
# print(feature.shape)

pca = PCA(n_components=0.99, whiten=True)

pca_fetures = pca.fit_transform(feature)

# print(feature[0].shape)
# for i in range(10):
#     print(pca_fetures[i].shape)

# reducing dimension for linearlt inseperably data 

features, _ = make_circles(n_samples= 1000, random_state=1,noise=0.01)
kpca = KernelPCA(kernel = 'rbf', gamma = 15, n_components=1)
kpca_features = kpca.fit_transform(features)
print(features.shape[1])
print(kpca_features.shape[1])
print(features.data)
# print(type(features))
