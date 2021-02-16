import pandas as pd 
import numpy as np 


def detect_outliers(x):
    q1, q3 = np.percentile(x, [25,75])
    iqr = q3 - q1
    lb = q1 - (iqr * 1.5)
    ub = q3 + (iqr * 1.5)
    return np.where((x> ub) | (x<lb))


houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

outliers = detect_outliers(houses['Square_Feet'])

print(houses)

print(outliers)