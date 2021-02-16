import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

features = [
            ("Texas", "Florida"),
            ("California", "Alabama"),
            ("Texas", "Florida"),
            ("Delware", "Florida"),
            ("Texas", "Alabama")
        ]


binarizer = MultiLabelBinarizer()
arr = binarizer.fit_transform(features)
print(binarizer.classes_)
print(arr)