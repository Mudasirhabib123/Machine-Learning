import numpy as np 


mat = np.array([[1,2,3],[4,5,6],[7,8,9]])

add_hundred = lambda x: x + 100

add_matrix = np.vectorize(add_hundred)

print(add_matrix(mat))