import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}

lista = [
    doc_1_word_count,doc_2_word_count,doc_3_word_count,doc_4_word_count
]

dictvectorizer = DictVectorizer(sparse = False)

matrix = dictvectorizer.fit_transform(list)

print(matrix)