### 12.11.2018 experiments

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url, names=names)
# Pregnancies - Number of times pregnant - Numeric
# Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test - Numeric
# BloodPressure - Diastolic blood pressure (mm Hg) - Numeric
# SkinThickness - Triceps skin fold thickness (mm) - Numeric
# Insulin - 2-Hour serum insulin (mu U/ml) - Numeric
# BMI - Body mass index (weight in kg/(height in m)^2) - Numeric
# DiabetesPedigreeFunction - Diabetes pedigree function - Numeric
# Age - Age (years) - Numeric
# Outcome - Class variable (0 or 1) - Numeric

#df.boxplot()
#df.hist()
#df.groupby('class').hist()
#df.groupby('class').plas.hist(alpha=0.4)
df.groupby('class').age.hist(alpha=0.4)
plt.show()

# функция для разделения датасета на обучающую и тестовую выборки
def stratified_split(y, proportion=0.8):
    y = np.array(y)

    train_inds = np.zeros(len(y), dtype=bool)
    test_inds = np.zeros(len(y), dtype=bool)

    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y == value)[0]
        np.random.shuffle(value_inds)

        n = int(proportion * len(value_inds))

        train_inds[value_inds[:n]] = True
        test_inds[value_inds[n:]] = True

    return train_inds, test_inds

# разделение датасета на обучающую и тестовую выборки
train, test = stratified_split(df['class'])

X_train = df.iloc[train, 0:8]
X_test = df.iloc[test, 0:8]

y_train = df['class'][train]
y_test = df['class'][test]

print(X_train.shape)
print(X_test.shape)

# импорт инструмента "Логистическая регрессия" для моделирования
from sklearn.linear_model import LogisticRegression

# построение модели
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# функция расчета точности модели
def accuracy(y_test, y_pred):
    return 1 - sum(abs(y_test - y_pred)/len(y_test))

print(accuracy(y_test, y_pred), ' RAW DATA')

# нормализация данных (для сравнения результатов моделирования на основе исходных данных)
def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()
    normalized = (arr - mean) / std
    return normalized


def norm_df(df):
    result = df.copy()
    for feature in df.columns:
        result[feature] = norm_arr(result[feature])
    return result

X_train = norm_df(df.iloc[train, 0:8])
X_test = norm_df(df.iloc[test, 0:8])
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(accuracy(y_test, y_pred), ' NORMALIZED DATA')

# импорт инструмента RANDOMFOREST (деревья решений) для моделирования
from sklearn.ensemble import RandomForestClassifier

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

# compare
X_train = df.iloc[train, 0:8]
X_test = df.iloc[test, 0:8]

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy(y_test, y_pred)

# vs
X_train = norm_df(df.iloc[train, 0:8])
X_test = norm_df(df.iloc[test, 0:8])

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy(y_test, y_pred)

# Cross-validation функция
def CV(df, classifier, nfold, norm=True):
    acc = []
    for i in range(nfold):
        y = df['class']
        train, test = stratified_split(y)

        if norm:
            X_train = norm_df(df.iloc[train, 0:8])
            X_test = norm_df(df.iloc[test, 0:8])
        else:
            X_train = df.iloc[train, 0:8]
            X_test = df.iloc[test, 0:8]

        y_train = y[train]
        y_test = y[test]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc.append(accuracy(y_test, y_pred))

    return acc


# classifier:
logreg = LogisticRegression()
rf = RandomForestClassifier()

res = CV(df, rf, 10, norm=False)
#res = np.array(res)
#print(res)
#print(res.mean())

##### HOMEWORK
#####  расчет точности для 4-х вариантов прогоноза

N = 10 # number of calculations
labels = ('logreg', 'rf', 'logreg norm', 'rf norm') # names for columns(used like pointers for models)
myDict = {} # dictionary to make DataFrame later

# how we choose what way we go to calculate an accuracy for each model:
for i in range (len(labels)):
    if 'logreg' in labels[i]:
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()
    if 'norm' in labels[i]:
        norm = True
    else:
        norm = False
    result = CV(df, model, N, norm=norm)
    myDict[labels[i]] = result

# results are placed in DataFrame for analysis
xs1 = pd.DataFrame(myDict)
print(xs1.describe())
#  but selections for each calculation are randomized!

#### what if we want to see results on the same selection to compare?

def myCV(df, label, nfold):
    myDict = {}
    for i in range(nfold):
        y = df['class']
        train, test = stratified_split(y)
        X_train = df.iloc[train, 0:8]
        X_test = df.iloc[test, 0:8]
# now selection is the same for each model
        for j in range (len(label)):
            if 'logreg' in label[j]:
                model = LogisticRegression()
            else:
                model = RandomForestClassifier()
            if 'norm' in label[j]:
                X_train = norm_df(X_train)
                X_test = norm_df(X_test)

            y_train = y[train]
            y_test = y[test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if label[j] not in myDict:
                myDict[label[j]] = list()
            myDict[label[j]].append(accuracy(y_test, y_pred))

    results = pd.DataFrame(myDict)

    return results

xs2 = myCV(df, labels, N)
print(xs2.describe())
