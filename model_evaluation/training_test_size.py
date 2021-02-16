import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import numpy as np 


digits = load_digits()
features, target = digits.data, digits.target


train_size, train_score, test_score = learning_curve(RandomForestClassifier(),features,target, n_jobs=-1, cv = 10, scoring='accuracy',train_sizes=np.linspace(
0.01,
1.0,
50))

train_mean = np.mean(train_score, axis= 1)
train_std = np.std(train_score, axis= 1)

test_mean = np.mean(test_score, axis= 1)
test_std = np.std(test_score, axis= 1)

plt.plot(train_size,train_mean, '--', label = 'training score')
plt.plot(train_size,test_mean, label = "Cross validaton score")


plt.show()
