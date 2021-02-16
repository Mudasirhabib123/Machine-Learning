from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784')

x,y = mnist.data,mnist.target
print(y[10])

digit=x[10]
digit=digit.reshape(28,28)

x_train,x_test=x[:60000],x[60000:]
y_train,y_test=x[:60000],x[60000:]

plt.imshow(digit , cmap = matplotlib.cm.binary , interpolation = 'nearest')
plt.show()

# print(x.shape)