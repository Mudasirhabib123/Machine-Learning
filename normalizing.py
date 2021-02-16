import numpy as np
from sklearn.preprocessing import Normalizer

# features = np.array([
#     [-23,-22],[-12,-13.23],[-2.345,-4,32],[1.21,1],[3.98,4.12]
# ])

features = np.array([[0.5, 0.5],
[1.1, 3.4],
[1.5, 20.2],
[1.63, 34.4],
[10.9, 3.3]])
# creating normalizer for default norm =l2
normalizer_l2 = Normalizer()
# creating normalizer for norm =l1
normalizer_l1 = Normalizer(norm = 'l1')

norm_l2_feature = normalizer_l2.transform(features)
norm_l1_feature = normalizer_l1.transform(features)

print(norm_l2_feature,'\n\n')
print(norm_l1_feature)
