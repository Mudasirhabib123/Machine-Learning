from sklearn.datasets import load_iris
import numpy as np 
from sklearn import tree


iris=load_iris()
print(iris.target_names)
removed=[22,34,56]
training_target = np.delete(iris.target,removed)
training_data = np.delete(iris.data,removed,axis=0)
print(training_data.data)
clf= tree.DecisionTreeClassifier()

clf.fit(training_data,training_target)

pred=clf.predict(iris.data[removed])
names=iris.target_names[pred]

print("prediction for iris=" ,names)