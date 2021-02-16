from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


boston = load_boston()
features = boston.data
target = boston.target
features_train, features_test, target_train, target_test = train_test_split(features, target)
dummy = DummyRegressor(strategy= 'mean')

dummy.fit(features_train, target_train)

print(dummy.score(features_test, target_test))

log_scl = LinearRegression()
log_scl.fit(features_train,target_train)
print('\n',log_scl.score(features_test, target_test))
