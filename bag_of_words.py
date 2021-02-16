from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 

text_data = np.array([
    'I love Brazil. Brazil!',
    'Sweden is best',
    'Germany beats both'
    ])


counter = CountVectorizer()
data = counter.fit_transform(text_data)

print(data.toarray())