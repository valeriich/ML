import pandas as pd
import math

df = pd.read_csv('tennis.csv')
print(df)

df['PlayTennis'].value_counts()

E_True =