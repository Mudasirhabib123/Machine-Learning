import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# array for training
x = np.array([
    [0, 2.10, 1.45],
    [1, 1.18, 1.33],
    [0, 1.22, 1.27],
    [1, -0.21, -1.19]
])  

# array having missing values to be predict
x_nan = np.array([
                        [np.nan, 0.87, 1.31],
                        [np.nan, -0.67, -0.22]
                    ])

# trainig model
model = KNeighborsClassifier(3,weights='distance')
model = model.fit(x[:,1:],x[:,0])

# predicted values
imputed = model.predict(x_nan[:,1:])

# stacking predicted to nan values
x_imputed = np.hstack((imputed.reshape(-1,1),x_nan[:,1:]))


complete_x = np.vstack((x_imputed,x))


print(complete_x)


# with imputer 
print('\n\n')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent') # by most frequent
# imputer = SimpleImputer(missing_values=np.nan,strategy='mean') # by mean however it may not work beacuse we need only boolean value if
# imputer = SimpleImputer(missing_values=np.nan,strategy='median') # by meadian however it also may not work beacuse we need only boolean value if

complete_x = np.vstack((x,x_nan))
complete_x = imputer.fit_transform(complete_x)
print (complete_x)