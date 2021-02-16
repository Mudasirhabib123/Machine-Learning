import pandas as pd
import numpy as np 

dates = np.array([
    '23-9-2016 12:25 PM',
    '3-3-2019 1:25 PM',
    '2-6-2015 12:5 pm'

])

dates = [pd.to_datetime(date, format="%d-%m-%Y %I:%M %p" , errors = 'coerce') for date in dates]
print(dates)
