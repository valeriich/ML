import pandas as pd

data = pd.read_csv('titanic.csv')
print(data.columns)
data.info()


def stratified_split(df, y, proportion=0.8):
    train, test = pd.DataFrame(), pd.DataFrame()
    labels = df[y].unique()
    for i in labels:
        indexes = df[df[y] == i]
        indexes_train = indexes.sample(frac=proportion)
        indexes_test = indexes.drop(indexes_train.index)
        train = train.append(indexes_train)
        test = test.append(indexes_test)

    return train, test


train_, test_ = stratified_split(data, 'Survived')
train_.info()
test_.info()
print(test_['Survived'].value_counts() / (test_['Survived'].value_counts() + train_['Survived'].value_counts()))
