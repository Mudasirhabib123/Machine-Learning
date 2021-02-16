import numpy as np
import pandas as pd
from sklearn import preprocessing


# feature = np.array([[-234.43],[12.35],[-500.34],[67],[233]])
feature = np.array([[-3],[-2],[1],[2],[3]])

# creating minMax Scaller
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
# creating standard scaller
std_scaler = preprocessing.StandardScaler()
# creating robust scaler
robust_scaler = preprocessing.RobustScaler()


min_max_feature = min_max_scaler.fit_transform(feature)
std_feature = std_scaler.fit_transform(feature)
rob_feature = robust_scaler.fit_transform(feature)

print('Min Max Feature \n',min_max_feature,'\n\n')
print('Standard Scaler Feature \n',std_feature,'\n\n')
print('Robust Scaler Feature \n',rob_feature,'\n\n')


