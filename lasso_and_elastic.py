import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(42)
n_samples, n_features = 50,100
x=np.random.randn(n_samples,n_features)
# print(x)
