import pandas as pd 
import numpy as np 


def uc(x):
    return x+x
   
time_index = pd.date_range('02/03/2019',periods = 10, freq = '3M')
df = pd.DataFrame(index = time_index)
df['Sale'] = np.random.randint(1,10,10)
df = df.resample('2W').mean()
print(df)

for s in df['Sale']:
    print(s)

df = df['Sale'].apply(uc)
print(df)