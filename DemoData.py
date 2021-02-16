import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


demo_data = np.random.randint(10,100, (20,4))
# print(demo_data)

model=MinMaxScaler()
data=model.fit_transform(demo_data)
# print(data)
df=pd.DataFrame(data=data,columns=['k1','k2','k3','labels'])
# print(df)

x=df[['k1','k2','k3']]
y=df['labels']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

svc=SVC(kernel='linear')
svc.fit(x_train,y_train)
pred=svc.predict(x_test)
print("prediction is =", pred)
