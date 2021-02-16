import cv2
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

image_full = cv2.imread('plane.jpg')
plt.imshow(image_full, cmap= 'gray')
image = image_full[50:-50,50:-50]
plt.show()
plt.imshow(image, cmap= 'gray')
plt.show()
print(image.shape)