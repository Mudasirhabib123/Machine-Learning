import numpy as np 
from scipy import sparse


mat  = np.array([[0,0,0],[0,0,1],[5,0,0]])
print(mat)
print('\n\n')

print(sparse.csr_matrix(mat))

