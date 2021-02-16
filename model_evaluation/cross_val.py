from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression


digits = load_digits()
features = digits['data']
target = digits['target']

features_train, features_test, target_train, target_test = train_test_split(features, target)
std_scl = StandardScaler()
log_scl = LogisticRegression()
kf = KFold(n_splits=10, shuffle=True, random_state= 1)

pipeline = make_pipeline(std_scl,log_scl)

cv_result = cross_val_score(pipeline, features, target, scoring= "accuracy", n_jobs= -1)

print(cv_result.mean())