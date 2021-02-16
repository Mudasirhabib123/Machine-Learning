from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np

dict = [
    {"Red":1,"Blue": 4},
    {"Red":2,"Blue": 3},
    {"Red":4,"Yellow": 2},
    {"Red":2,"Yellow": 2}
]

dictvectorizer = DictVectorizer(sparse = False)

feature = dictvectorizer.fit_transform(dict)
# printing result
# print(dictvectorizer.get_feature_names())
# print(feature)

# making dataframe for better result view

df = pd.DataFrame(feature, columns = dictvectorizer.get_feature_names())
print(df)