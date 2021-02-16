import numpy as np
import pandas as pd


df = pd.DataFrame({'Score' : ['low','low','medium','high']})

scaller = {'low' : 1 ,'medium' : 2, 'high' : 3}

df = df['Score'].replace(scaller)

print(df)