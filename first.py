import numpy as nm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import neighbors
import matplotlib.pyplot as plt
x = nm.random.random((10, 5))
y = nm.array(['F', 'M', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F'])
x_tran, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
svc = SVC(kernel='linear')
knn = neighbors.KNeighborsClassifier(n_neighbors=2)
knn.fit(x_tran, y_train)
knn_pred = knn.predict(x_tran)
svc.fit(x_tran, y_train)
pred = svc.predict(x_tran)
print(pred)
print(knn_pred)

# plt.scatter(x_test,y_test)
# plt.show()