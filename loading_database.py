import pandas as pd 


df = pd.read_csv('titanic.csv')
# df = pd.read_csv('https://tinyurl.com/titanic-csv')
print(df.count())
print('\n\n')
df['Sex'] = df['Sex'].replace(['female', 'male'], ['Woman', 'man'])
print(df.head())
print('\n\n')
df = df.drop('SexCode',axis = 1)
print(df.head())
print('\n\n')
df = df.drop_duplicates(subset= ['Age'], keep = 'last')
print(df)
print(df.count())
