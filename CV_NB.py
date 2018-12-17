import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

def gaussian(value, mu, sigma):
    res = 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(value-mu)**2/(2*sigma**2))
    return res


def runner():
    # Importing dataset
    data = pd.read_csv("train.csv")

    # Convert categorical variable to numeric
    data["Sex_cleaned"] = np.where(data["Sex"] == "male", 0, 1)
    data["Embarked_cleaned"] = np.where(data["Embarked"] == "S", 0,
                                        np.where(data["Embarked"] == "C", 1, np.where(data["Embarked"] == "Q", 2, 3)))

    # Cleaning dataset of NaN
    data = data[[
        "Survived",
        "Pclass",
        "Sex_cleaned",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_cleaned"
    ]].dropna(axis=0, how='any')

    # Split dataset in training and test datasets
    X_train, X_test = train_test_split(data, test_size=0.3)

    gnb = BernoulliNB()
    used_features =[
        "Pclass",
        "Sex_cleaned",
        "Age",
        "SibSp",
#        "Parch",
#        "Fare",
#        "Embarked_cleaned"
    ]

    # Train classifier
    gnb.fit(
        X_train[used_features].values,
        X_train["Survived"]
    )
    y_pred = gnb.predict(X_test[used_features])

    # Print results
    print("Number of mislabeled points out of a total {} points : {}, performance {}"
          .format(
              X_test.shape[0],
              (X_test["Survived"] != y_pred).sum(),
              100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
    ))

    return 100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])


def cycle():
    y = 0
    for x in range(1, 11):
        y += runner()
    print(y/10)

cycle()