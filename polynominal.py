import numpy as np
from sklearn.preprocessing import PolynomialFeatures

feature = np.array([
    [2,3],[2,4]
])

interaction = PolynomialFeatures(degree = 2,include_bias = False)
poly_feature = interaction.fit_transform(feature)

print(poly_feature)

