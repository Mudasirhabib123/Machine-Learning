import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes= datasets.load_diabetes()
# print(diabetes.keys())
diabetes_x=diabetes.data[:,np.newaxis,2]
# diabetes_x=diabetes.data

# print(diabetes_x)
diabetes_x_train=diabetes_x[:-30]
diabetes_x_test=diabetes_x[-20:]


diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes.target[-20:]

model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)

diabetes_y_pred = model.predict(diabetes_x_test)
print("Mean squad: ",mean_squared_error(diabetes_y_test,diabetes_y_pred))
# print("Weight : ",model.coef_)
print("Intercept : ",model.intercept_)

plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_pred)

plt.show()
