from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


x, y = make_classification(n_samples= 1000, n_features= 3,n_informative=3, random_state= 1, n_redundant=0)

log_clf = LogisticRegression()

score = cross_val_score(log_clf, x, y, scoring= 'accuracy')

print(score)