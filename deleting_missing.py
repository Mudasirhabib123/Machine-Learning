# deleting nan values
# with numpy

import numpy as np

features = np.array([[1.1, 11.1],
[2.2, 22.2],
[3.3, 33.3],
[4.4, 44.4],
[np.nan, 55]])


indexes = [~np.isnan(features).any(axis = 1)]
features = features[indexes]
print(features)

# with pandas

import pandas as pd

features = np.array([[1.1, 11.1],
[2.2, 22.2],
[3.3, 33.3],
[4.4, 44.4],
[np.nan, 55]])

df = pd.DataFrame(features, columns = ['featuren 1', 'feature 2'])
df = df.dropna()
print(df)