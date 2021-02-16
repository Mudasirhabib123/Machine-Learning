import numpy as np 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


iris = load_iris()
features, target = iris.data, iris.target
logistic = LogisticRegression()
penality = ['l1', 'l2']
c= np.logspace(0,4,10)
gridcv = GridSearchCV(logistic, dict(C = c, penalty = penality), cv = 5, verbose= 0)

best_model = gridcv.fit(features, target)

print(best_model.best_estimator_.get_params()['penalty'])