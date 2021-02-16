import pandas as pd 
import numpy as numpy


timestamp = pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')
date = pd.Timestamp('2017-05-01 06:00:00')
london_date = date.tz_localize('Asia/karachi')
london_date.tz_convert('europe/london')
print(london_date)




# print(timestamp.now())