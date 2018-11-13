import pandas as pd
import numpy as np


def norm_df(df):
    result = df.cpopy()
    for feature in df.columns:
        result[feature] = norm_arr(result[feature])

    return result


def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized


df = pd.read_csv('titanic.csv')
df['Age']
df['Age'].isnull().values
df1 = df[df['Age'].notnull()]
print(norm_arr(df1['Age']))
df1['Age'].iloc[0:3] = 1000
print(norm_arr(df1['Age']))
# удалить строки с крайними 2,5% значениями в столбце 'Age'
selection = (df1['Age'] < df1.Age.quantile(0.975)) & (df1['Age'] > df1.Age.quantile(0.025))
print(selection)
selected = df1[selection]
print(selected['Age'])
print(norm_arr(selected['Age']))
