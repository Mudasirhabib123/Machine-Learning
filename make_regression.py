from sklearn.datasets import make_regression

features, target = make_regression(n_samples= 100, n_targets= 1,n_features= 2)

print(len(features[: , :]))