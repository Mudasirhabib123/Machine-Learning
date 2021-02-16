import pandas as pd
import numpy as np


titanic = pd.read_csv('titanic.csv')
# print(titanic.head(2))

dataframe = pd.DataFrame()
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]
new_person = pd.Series(['moley',32, True],index = ['Name','Age','Driver'])
dataframe = dataframe.append(new_person, ignore_index = True)
new_person = pd.Series(['jakson',33, True],index = ['Name','Age','Driver'])
dataframe = dataframe.append(new_person, ignore_index = True)

data = (dataframe['Driver']==True).head(1)

print(data)



