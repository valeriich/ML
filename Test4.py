### 19.11.2018 Decision trees

import pandas as pd
import math

df = pd.read_csv('Tennis.csv')


def entropy(N, n):
    sum = 0
    for i in range(len(n)):
        if n[i] != 0:
            sum += - n[i] / N * math.log(n[i] / N, 2)
    return sum


n = (df['PlayTennis'].value_counts()['Yes'], df['PlayTennis'].value_counts()['No'])
print(n)
N_tree = len(df)
print(N_tree)
# Энтропия дерева
E = entropy(N_tree, n)
print('%.3f' % E)

# Энтропия признака "Windy"
n_windy_true = (len(df[(df['Windy'] == True) & (df['PlayTennis'] == "Yes")]),
                len(df[(df['Windy'] == True) & (df['PlayTennis'] == "No")]))
print(n_windy_true)
N_windy_true = int(df['Windy'].value_counts()[True])
print(N_windy_true)
E_windy_true = float(entropy(N_windy_true, n_windy_true))
print('%.3f' % E_windy_true)

n_windy_false = (len(df[(df['Windy'] == False) & (df['PlayTennis'] == "Yes")]),
                 len(df[(df['Windy'] == False) & (df['PlayTennis'] == "No")]))
print(n_windy_false)
N_windy_false = df['Windy'].value_counts()[False]
print(N_windy_false)
E_windy_false = float(entropy(N_windy_false, n_windy_false))
print('%.3f' % E_windy_false)

E_windy = E_windy_true * (N_windy_true / N_tree) + E_windy_false * (N_windy_false / N_tree)
print('Энтропия WINDY: ', '%.3f' % E_windy)
IG_windy = E - E_windy
print('Information gain WINDY: ', '%.3f' % IG_windy)

# Энтропия признака "Humidity"
n_Humidity_high = (len(df[(df['Humidity'] == 'High') & (df['PlayTennis'] == "Yes")]),
                   len(df[(df['Humidity'] == 'High') & (df['PlayTennis'] == "No")]))
print(n_Humidity_high)
N_Humidity_high = int(df['Humidity'].value_counts()['High'])
print(N_Humidity_high)
E_Humidity_high = float(entropy(N_Humidity_high, n_Humidity_high))
print('%.3f' % E_Humidity_high)

n_Humidity_normal = (len(df[(df['Humidity'] == 'Normal') & (df['PlayTennis'] == "Yes")]),
                     len(df[(df['Humidity'] == 'Normal') & (df['PlayTennis'] == "No")]))
print(n_Humidity_normal)
N_Humidity_normal = df['Humidity'].value_counts()['Normal']
print(N_Humidity_normal)
E_Humidity_normal = float(entropy(N_Humidity_normal, n_Humidity_normal))
print('%.3f' % E_Humidity_normal)

E_Humidity = E_Humidity_high * (N_Humidity_high / N_tree) + E_Humidity_normal * (N_Humidity_normal / N_tree)
print('Энтропия HUMIDITY: ', '%.3f' % E_Humidity)
IG_Humidity = E - E_Humidity
print('Information gain HUMIDITY: ', '%.3f' % IG_Humidity)

# Энтропия признака "Temperature"
n_Temperature_hot = (len(df[(df['Temperature'] == 'Hot') & (df['PlayTennis'] == "Yes")]),
                     len(df[(df['Temperature'] == 'Hot') & (df['PlayTennis'] == "No")]))
print(n_Temperature_hot)
N_Temperature_hot = int(df['Temperature'].value_counts()['Hot'])
print(N_Temperature_hot)
E_Temperature_hot = float(entropy(N_Temperature_hot, n_Temperature_hot))
print('%.3f' % E_Temperature_hot)

n_Temperature_mild = (len(df[(df['Temperature'] == 'Mild') & (df['PlayTennis'] == "Yes")]),
                      len(df[(df['Temperature'] == 'Mild') & (df['PlayTennis'] == "No")]))
print(n_Temperature_mild)
N_Temperature_mild = df['Temperature'].value_counts()['Mild']
print(N_Temperature_mild)
E_Temperature_mild = float(entropy(N_Temperature_mild, n_Temperature_mild))
print('%.3f' % E_Temperature_mild)

n_Temperature_cold = (len(df[(df['Temperature'] == 'Cool') & (df['PlayTennis'] == "Yes")]),
                      len(df[(df['Temperature'] == 'Cool') & (df['PlayTennis'] == "No")]))
print(n_Temperature_cold)
N_Temperature_cold = df['Temperature'].value_counts()['Cool']
print(N_Temperature_cold)
E_Temperature_cold = entropy(N_Temperature_cold, n_Temperature_cold)
print('%.3f' % E_Temperature_cold)

E_Temperature = E_Temperature_hot * (N_Temperature_hot / N_tree) + E_Temperature_mild * (
            N_Temperature_mild / N_tree) + E_Temperature_cold * (N_Temperature_cold / N_tree)
print('Энтропия Temperature: ', '%.3f' % E_Temperature)
IG_Temperature = E - E_Temperature
print('Information gain Temperature: ', '%.3f' % IG_Temperature)

# Энтропия признака 'Outlook'
n_Outlook_Sunny = (len(df[(df['Outlook'] == 'Sunny') & (df['PlayTennis'] == "Yes")]),
                   len(df[(df['Outlook'] == 'Sunny') & (df['PlayTennis'] == "No")]))
print(n_Outlook_Sunny)
N_Outlook_Sunny = int(df['Outlook'].value_counts()['Sunny'])
print(N_Outlook_Sunny)
E_Outlook_Sunny = float(entropy(N_Outlook_Sunny, n_Outlook_Sunny))
print('%.3f' % E_Outlook_Sunny)

n_Outlook_Overcast = (len(df[(df['Outlook'] == 'Overcast') & (df['PlayTennis'] == "Yes")]),
                      len(df[(df['Outlook'] == 'Overcast') & (df['PlayTennis'] == "No")]))
print(n_Outlook_Overcast)
N_Outlook_Overcast = df['Outlook'].value_counts()['Overcast']
print(N_Outlook_Overcast)
E_Outlook_Overcast = entropy(N_Outlook_Overcast, n_Outlook_Overcast)
print('%.3f' % E_Outlook_Overcast)

n_Outlook_Rainy = (len(df[(df['Outlook'] == 'Rainy') & (df['PlayTennis'] == "Yes")]),
                   len(df[(df['Outlook'] == 'Rainy') & (df['PlayTennis'] == "No")]))
print(n_Outlook_Rainy)
N_Outlook_Rainy = df['Outlook'].value_counts()['Rainy']
print(N_Outlook_Rainy)
E_Outlook_Rainy = entropy(N_Outlook_Rainy, n_Outlook_Rainy)
print('%.3f' % E_Outlook_Rainy)

E_Outlook = E_Outlook_Sunny * (N_Outlook_Sunny / N_tree) + E_Outlook_Overcast * (
            N_Outlook_Overcast / N_tree) + E_Outlook_Rainy * (N_Outlook_Rainy / N_tree)
print('Энтропия Outlook: ', '%.3f' % E_Outlook)
IG_Outlook = E - E_Outlook
print('Information gain Outlook: ', '%.3f' % IG_Outlook)
