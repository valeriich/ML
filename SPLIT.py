import pandas as pd
import numpy as np

data = pd.read_csv('titanic.csv')
print(data.columns)

# разделение данных по признаку "выжившие"
Yes = data[data['Survived'] == 1]
Not = data[data['Survived'] == 0]
print(Yes.info())

# разделение данных в признаке "выжившие"
x_train_y = Yes.sample(frac=0.8)
x_test_y = Yes.drop(x_train_y.index)
y_train_y = x_train_y['Survived']
x_train_y = x_train_y.drop(['Survived'], axis=1)
y_test_y = x_test_y['Survived']
x_test_y = x_test_y.drop(['Survived'], axis=1)

print(x_train_y.columns)
print(y_train_y)

# разделение данных в признаке " не выжившие"
x_train_n = Not.sample(frac=0.8)
x_test_n = Not.drop(x_train_n.index)
print(x_test_n)
y_train_n = x_train_n['Survived']
x_train_n = x_train_n.drop(['Survived'], axis=1)
y_test_n = x_test_n['Survived']
x_test_n = x_test_n.drop(['Survived'], axis=1)
print(y_test_n)

# объединение данных
x_train = pd.concat([x_train_y, x_train_n])
print(x_train.info())
y_train = pd.concat([y_train_y, y_train_n])
print(len(y_train))