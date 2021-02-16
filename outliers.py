import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

def indecise_of_outliers(x):
    q1,q3 = np.percentile(x,[25, 75])
    iqr = q3 - q1  
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x>upper_bound) | (x<lower_bound))


features, _ = make_blobs(n_samples = 10, n_features = 2, random_state = 1, centers = 1)

# print(features)
features[3,0] =100
features[3,1] =100

outlier_detector = EllipticEnvelope(contamination = .1)
outlier_detector.fit(features)
res = outlier_detector.predict(features)
print(res)

res  =indecise_of_outliers(features[:,0])
print(features[res])