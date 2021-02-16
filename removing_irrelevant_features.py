from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2,f_classif


iris = load_iris()

features = iris.data
target = iris.target
# features = features.astype(int)
kbest = SelectKBest(chi2, k=2)
kb_features = kbest.fit_transform(features, target)

print("orogional features", features.shape[1])
print("k best features", kb_features.shape[1])

kbest = SelectKBest(f_classif, k=1)
kb_features = kbest.fit_transform(features, target)
print('\n')
print("k best features", kb_features.shape[1])

spbest = SelectPercentile(f_classif, percentile= 75)
sp_features = spbest.fit_transform(features, target)

print('\n')
print("sp best features", sp_features.shape[1])
