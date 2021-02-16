from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


iris=datasets.load_iris()
x=iris.data[:,3:]
xn=np.linspace(0,3,1000).reshape(-1,1)

y=(iris.target==2).astype(np.int)
lin_model=LinearRegression()
log_model=LogisticRegression()
lin_model.fit(x,y)
log_model.fit(x,y)
y_prob=log_model.predict_proba(xn)
print(y_prob)
plt.plot(xn,y_prob,'g-',label='verginica')
# print("mea                                                                                                    n squared error: ", mean_squared_error(xn,y_prob))
plin=lin_model.predict(([[1.62]]))
plog=log_model.predict(([[1.62]]))
plt.show()

print(plin)
print(plog)
