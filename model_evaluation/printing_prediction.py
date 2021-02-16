from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


iris = load_iris()
features = iris.data
target = iris.target
features_train, features_test, target_train, target_test = train_test_split(features, target)

model = LogisticRegression()
model.fit(features_train,target_train)

target_predicted = model.predict(features_test)

print(classification_report(target_test, target_predicted))