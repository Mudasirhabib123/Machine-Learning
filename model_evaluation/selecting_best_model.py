from sklearn.datasets import load_iris
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


iris = load_iris()
features = iris.data
target = iris.target

logistic = LogisticRegression()

hyperparameter = dict(C =uniform(loc = 0, scale = 4),penalty = ['l1','l2'])

randomize = RandomizedSearchCV(logistic, hyperparameter,random_state= 1 ,n_iter=100, cv=5, verbose=0)

best_model = randomize.fit(features, target)
best_model_predicted = bm.predict(features)
print(best_model_predicted)
