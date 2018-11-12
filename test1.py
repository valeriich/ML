import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('apple.csv', index_col='Date', parse_dates=True)
df = df.sort_index()
print(df.info())
df.loc['2012-Feb', 'Close'].mean()
print(df.loc['2012-Feb':'2015-Feb', 'Close'].mean())
print(df[:4])

# grouping by periods
print(df.resample('W')['Close'].mean())
montly_average = df.resample('M')['Close'].mean()
montly_average.columns = ['Date', 'Average Price']
new_sample_df = df.loc['2012-Feb':'2017-Feb', ['Close']]
new_sample_df.plot()
plt.show()

# year average output
s = df.resample('Y')['Close'].mean()
print(pd.DataFrame({'Average': s.values, 'Period': s.index}))
