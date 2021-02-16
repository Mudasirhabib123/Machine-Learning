from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split



iris = load_iris()
features = iris.data
target = iris.target
features_train, features_test, target_train, target_test = train_test_split(features, target)
clf = LogisticRegression()

predicted_target = clf.fit(features_train,target_train).predict(features_test)

matrix = confusion_matrix(target_test, predicted_target)

df = DataFrame(matrix, index = iris.target_names, columns = iris.target_names)

sns.heatmap(df, annot = True, cbar = None, cmap = 'Blues')
plt.show()