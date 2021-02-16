from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,make_scorer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier


features, target = make_regression(n_samples=100, n_features=3)

features_train, features_test, target_train, target_test = train_test_split(features, target)

def custom_scorer(taeget_test, targte_predicted):
    return r2_score(target_test,targte_predicted)

score = make_scorer(custom_scorer,greater_is_better=True)

# model = Ridge()
model = RandomForestClassifier()
model.fit(features_train,target_train)
target_predicted = model.predict(features_test)
print(target_predicted)

score = score(model,features_test, target_test)

print('Score',score)