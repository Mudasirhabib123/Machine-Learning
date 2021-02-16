import numpy as np

from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Imputer
from sklearn.datasets import make_blobs
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

features, _ = make_blobs(n_samples = 100, n_features = 2, random_state = 1)
std_features = StandardScaler().fit_transform(features)
actual_val = std_features[0,0]
mean_imputer = imputer.fit_transform(std_features)

print('actual value', actual_val,'\n\n')
print('predicted value', mean_imputer[0,0],'\n\n')
