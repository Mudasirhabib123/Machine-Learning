import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from fancyimpute import KNN # using another library fancy impute

features, _ = make_blobs(n_samples = 100, n_features = 2, random_state = 1) 
scaler = StandardScaler()

std_features = scaler.fit_transform(features)

actual_value = std_features[0,0]
std_feature[0,0] = np.nan

knn_imputed = KNN(k=5, verbose = 0).complete(std_features)

print(actual_value,'\n')
print(knn_imputed[0,0],'\n')

