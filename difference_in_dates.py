import pandas as pd 
import numpy as np 

dates = pd.DataFrame()
dates['Arival'] = [pd.Timestamp('1-1-2017'), pd.Timestamp('1-2-2017')]
dates['Left'] = [pd.Timestamp('4-1-2017'), pd.Timestamp('1-3-2017')]
dates['Difference'] = dates['Left'] - dates['Arival']

print(dates)

series = pd.Series([d.days for d in (dates['Left'] - dates['Arival'])])


print(series)