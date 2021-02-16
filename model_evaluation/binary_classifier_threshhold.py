from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 



features, target = make_classification(n_samples=10000, n_features=3, n_informative= 3, n_redundant=0, random_state=1)

features_train, features_test, target_train, target_test = train_test_split(features, target)

log_clf = LogisticRegression()
log_clf.fit(features_train, target_train)

target_proba = log_clf.predict_proba(features_test)[:,1]

fp_rate, tp_rate, threshhold = roc_curve(target_test,target_proba)

plt.title("Receving Operating Curve")
plt.xlabel('FP Rate')
plt.ylabel('TP Rate')
plt.plot(fp_rate,tp_rate)
plt.plot([0,1], ls= '--')
plt.plot([0,0],[1,0],[1,1])
plt.show()

