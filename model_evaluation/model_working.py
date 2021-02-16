from sklearn import datasets
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


digits = datasets.load_digits()
features = digits.data
target = digits.target

std_scl = StandardScaler()
log_reg = LogisticRegression()

pipeline = make_pipeline(std_scl, log_reg)

kf = KFold(n_splits=10, shuffle=True, random_state= 1)

cv_result = cross_val_score(pipeline, features,target, cv = kf, scoring="accuracy",n_jobs=-1)

# print(cv_result.mean())


