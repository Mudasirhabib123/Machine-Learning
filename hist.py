import cv2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.Series([1, 1, 1, 2, 2, 3, 3, 3, 4, 5])

data.hist(grid = False)
image = cv2.imread('plane.jpg',cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

colors = ('r','g','b')

for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image], [i],None,  [256], [0,256])
    plt.plot(histogram, color = channel)
    plt.xlim(0,256)

plt.show()