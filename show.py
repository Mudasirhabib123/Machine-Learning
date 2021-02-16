import matplotlib.pyplot as plt
from sklearn import datasets
print(__doc__)


digits = datasets.load_digits()

plt.figure(figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
