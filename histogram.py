import pandas as pd
import cv2
import numpy as np 
import matplotlib.pyplot as plt 


image = cv2.imread('plane.jpg',cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

features = []
colors = ('r','g','b')

for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image],[i],None, [256], [0,256])
    features.extend(histogram)

observation = np.array(features).flatten()
print(observation[:5])