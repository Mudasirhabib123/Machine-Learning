import pandas as pd 
import numpy as np 

df = pd.DataFrame()

df['date'] = pd.date_range('1/1/2001',periods= 10000, freq = 'H')

df = df[(df['date'] > '2001-01-01 00:00:00') & (df['date']< '2001-01-01 07:00:00')]

# df = df.set_index(df['date'])

# df = df.loc['2001-01-01 00:00:00':'2001-01-01 07:00:00']



# creating features from dataframe/
dff = pd.DataFrame()
dff['year'] = df['date'].dt.year
dff['month'] = df['date'].dt.month
dff['day'] = df['date'].dt.day
dff['hour'] = df['date'].dt.hour
dff['minute'] = df['date'].dt.minute


print(dff)
