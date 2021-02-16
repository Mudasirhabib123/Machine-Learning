import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
features = iris['data'][40:,:]
target = iris['target'][40:]

# print(target)

target = (np.where((target == 0 ), 0,1))

# print(target)

weights = {0: .9, 1: 0.1}
RandomForestClassifier(class_weight=weights)
RandomForestClassifier(class_weight='balanced')

index_iris = np.where(target == 0 )[0]
index_others = np.where(target == 1 )[0]

len_iris = len(index_iris)
len_others = len(index_others)

index_others_downsampled = np.random.choice(index_others, size=len_iris, replace=False)

target_20 = np.hstack((target[index_iris], target[index_others_downsampled]))
features_20 = np.vstack((features[index_iris,:],features[index_others_downsampled,:]))

index_iris_upsampled = np.random.choice(index_iris,len_others,replace=True)
target_200 = np.concatenate((target[index_iris_upsampled], target[index_others]))
# print(target_200)
features_200 = np.vstack((features[index_iris_upsampled,:],features[index_others]))
print (features_200)
# print(b)
# print(i_class1)