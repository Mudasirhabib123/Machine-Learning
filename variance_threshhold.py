from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


iris = load_iris()
features = iris['data']

vth = VarianceThreshold(threshold=0.5)
vth_features = vth.fit_transform(features)
# inversing features 
inv_features = vth.inverse_transform(vth_features)

# printing features 
print(features.shape)
print(vth_features.shape)
print(inv_features.shape)

# standardizing features 
scaler = StandardScaler()
std_features = scaler.fit_transform(features)
print(std_features.shape)
selector = VarianceThreshold()
selector.fit(std_features)
print(selector.variances_)
print('\n\n',vth.fit(features).variances_)