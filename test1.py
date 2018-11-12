import pandas as pd
import numpy as np

#my_series = pd.Series([5, 6, 7, 8, 9, 10])
#print(my_series)
#print(my_series.index)

#df = pd.read_csv('titanic.csv')
#df['oddish'] = np.where(df.PassengerID % 2 == 0, 'even', 'odd')
#print(df)

df = pd.read_csv('apple.csv', index_col='Date', parse_dates=True)
df = df.sort_index()
#print(df.info())
s = df.resample('Y')['Close'].mean()
#print(s)
#print(s.index)
#s = pd.Series([1, 2, 3, 4, 5], ['a', 'b', 'c', 'd', 'e'])
#a = pd.DataFrame(data = s.values, index = s.index)
a = pd.DataFrame({'Average': s.values, 'Period': s.index})
print(a)