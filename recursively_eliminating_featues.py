import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

warnings.filterwarnings(action = 'ignore', module= 'scipy', message= "^internal gelsd")
features, target = make_regression(n_samples=1000, n_features=100, n_informative=2, random_state= 1)

# print(features[:5,:2])

lin_model = linear_model.LinearRegression()

rf = RFECV(estimator = lin_model, step=1, scoring='neg_mean_squared_error')

rf.fit(features, target)
rf_features = rf.transform(features)

print(rf.n_features_)
print(rf.support_)
print(rf.ranking_)
