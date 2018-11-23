import pandas as pd
import numpy as np


def norm_df(df):
    result = df.copy()
    for feature in df.columns:
        result = result.loc[lambda df: (df[feature] < df[feature].quantile(0.975))& (df[feature] > df[feature].quantile(0.025)), :]
        result[feature] = norm_arr(result[feature])
    return result


def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()
    normalized = (arr - mean) / std
    return normalized


def select_df(df, feature):
    selection = (df[feature] < df[feature].quantile(0.975)) & (df1[feature] > df1[feature].quantile(0.025))
    result = df[selection]
    return result


df = pd.read_csv('titanic.csv')
df['Age']
df['Age'].isnull().values
df1 = df[df['Age'].notnull()]
#print(norm_arr(df1['Age']))
#df1['Age'].iloc[0:3] = 1000
#print(norm_arr(df1['Age']))

### удалить строки с крайними 2,5% значениями в столбце 'Age'
#selection = (df1['Age'] < df1.Age.quantile(0.975)) & (df1['Age'] > df1.Age.quantile(0.025))
#print(selection)
#selected = df1[selection]
#print(selected['Age'])

## второй вариант
##print(df1.loc[lambda df: (df.Age < df.Age.quantile(0.975))& (df.Age > df.Age.quantile(0.025)), :])

#print(norm_arr(selected['Age']))
#print(select_df(df1, 'Age'))

data = pd.read_csv('weather.csv')
print(data.columns)
print(data.shape)
data1 = norm_df(data[['TEMP', 'PRESSURE']])
print(data1)
data.insert(loc=len(data.columns), column='TEMP_norm', value=data1['TEMP'])
data.insert(loc=len(data.columns), column='PRESSURE_norm', value=data1['PRESSURE'])
print(data.head())
