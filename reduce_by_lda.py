from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

iris = load_iris()
features = iris['data']
target = iris['target']
lda = LinearDiscriminantAnalysis(n_components = 1)

lda_features = lda.fit_transform(features,target)
# lda_features = lda.transform(features)
print(features.shape)
print(lda_features.shape)
print(lda_features)
print(lda.explained_variance_ratio_)

