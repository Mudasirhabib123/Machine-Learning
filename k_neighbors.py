import numpy as np
from sklearn import datasets,neighbors,preprocessing


houses = datasets.load_boston()
# print(houses.feature_names)
data = houses.data
tar=houses.target
# transforming tartget data tar to be fit in fitting model
label_encoding=preprocessing.LabelEncoder()
tar_encoded=label_encoding.fit_transform(tar)
print(tar_encoded)
# knn = neighbors.KNeighborsClassifier()
# knn.fit(data,tar_encoded)
# pred=knn.predict([[1,1,1,1,11,11,1,1,1,1,1,1,1]])
# print(pred)
