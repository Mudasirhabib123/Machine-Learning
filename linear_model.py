from sklearn import datasets,linear_model
import numpy as np

x=np.array([[1,2,3]])
y=np.array([[3,2,4]])
p=np.array([[1,7,3]])
model=linear_model.LinearRegression()
model.fit(x,y)
pred=model.predict([[1,23,3]])
print(pred)