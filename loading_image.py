import pandas as pd 
import numpy as np 
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('plane.jpg',cv2.IMREAD_COLOR)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image,(50,50)) # resizing image
plt.imshow(image_rgb,cmap='gray')
plt.axis('off')
plt.show()

# print(type(image))

print(image[110,110])