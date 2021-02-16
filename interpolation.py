import pandas as pd 
import numpy as np 

df = pd.DataFrame()
df['Time'] = pd.date_range('1/1/2020',periods = 5, freq = 'M')

df['Sales'] = [1,np.nan,np.nan,np.nan,5]
# df = df.interpolate(limit = 2)
# df = df.bfill(limit = 2)  # or we can alse use ffill() for first value

print(df)