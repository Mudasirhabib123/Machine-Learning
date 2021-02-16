import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 


houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# print(houses)
# print(houses['Bathrooms'] <20)
# making outliers
houses['Outliers'] = np.where(houses['Bathrooms']<20 , 0, 1)
houses['Log_of_square_feet'] = [np.log(x) for x in houses["Square_Feet"]]

index = houses[houses['Outliers'] == 1].Bathrooms
imputer = SimpleImputer(missing_values = float(index), strategy = 'mean')
imputer.fit(houses)
houses = imputer.transform(houses)

print(houses)