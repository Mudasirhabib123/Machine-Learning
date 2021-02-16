from sklearn import datasets
from sklearn import neighbors
import numpy as np


iris=datasets.load_iris()

data=iris.data

tar = iris.target
# tar = np.linspace(1,3,150)
# print(tar)
knn = neighbors.KNeighborsClassifier()
knn.fit(data,tar)
pred = knn.predict([[1,5,1,1]])
print(pred)