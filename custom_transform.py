import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def add_ten(x):
    return x+10

features = np.array([
    [2,3],[2,3],[2,3]
])

transformer = FunctionTransformer(add_ten)

updated_features = transformer.fit_transform(features)

print(updated_features)

# using pandas library
df = pd.DataFrame(features, columns = ['feature1', 'feature2'])
df = df.apply(add_ten)
print(df)

