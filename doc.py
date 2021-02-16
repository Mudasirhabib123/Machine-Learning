from sklearn import datasets,svm,metrics
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
digit = datasets.load_digits()

model = svm.SVC(gamma= 0.001,C=100.)

model.fit(digit.data[:-1],digit.target[:-1])
res = model.predict(digit.data[-1:])
print(res)

dig = digit.data[-1:].reshape(8,8)

plt.imshow(dig,cmap=matplotlib.cm.binary)
plt.show()
 


# dig=digit.data[1]
# img=dig.reshape(8,8)
# print(digit.target[15])
# plt.imshow(img, cmap = matplotlib.cm.binary )
# plt.show()