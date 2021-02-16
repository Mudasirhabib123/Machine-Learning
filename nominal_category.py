import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

feature = np.array([
    ['Bahawalpur'],['Lahore'],['Karachi'],['Multan'],['Bahawalpur'],['Islamabad']
])

binarizer = LabelBinarizer()
arr = binarizer.fit_transform(feature)
# print(binarizer.classes_)
# print(arr)

# making dataframe for classes and arr
df = pd.DataFrame(arr, columns = [binarizer.classes_])
print(df)

# printing cities
inversed_arr = binarizer.inverse_transform(arr)
print(inversed_arr)


# getting same result by pandas
res = pd.get_dummies(feature[:,0])
# print(res)