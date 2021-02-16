from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


iris = load_iris()
features = iris.data
target = iris.target


dummy = DummyClassifier()
rnd_clf = RandomForestClassifier()

features_train, features_test, target_train, target_test = train_test_split(features, target, random_state = 0)

dummy.fit(features_train,target_train)
rnd_clf.fit(features_train, target_train)

print('Dummy', dummy.score(features_test, target_test))
print('Rando Forest classifier', rnd_clf.score(features_test, target_test))

