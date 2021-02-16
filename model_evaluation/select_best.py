import numpy as np 
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


iris = load_iris()
features = iris.data
target = iris.target

pipeline = make_pipeline(['classifier', RandomForestClassifier()])

search_space = [{
    'classifier': [LogisticRegression()],
    'classifier_penalty':np.logspace(0,4,10)
}]