import numpy as np
from sklearn.preprocessing import Binarizer

age = np.array([
    [12],[14],[20],[33],[55]
])

bin = Binarizer(18)

res = bin.fit_transform(age)
print(res)

res = np.digitize(age, bins = [18,28])  #not binary 
print(res)