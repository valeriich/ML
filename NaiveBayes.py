import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

def gaussian(value, mu, sigma):
    res = 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(value-mu)**2/(2*sigma**2))
    return res


# Importing dataset
data = pd.read_csv("train.csv")

# Convert categorical variable to numeric
data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,np.where(data["Embarked"]=="C",1, np.where(data["Embarked"]=="Q",2,3)))

# Cleaning dataset of NaN
data=data[[
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
X_train, X_test = train_test_split(data, test_size=0.3, random_state=int(time.time()*2))

gnb = BernoulliNB()
used_features =[
    "Pclass",
    "Sex_cleaned",
#    "Age",
#    "SibSp",
#    "Parch",
#    "Fare",
#    "Embarked_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("BernoulliNB - Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))


mean_fare_survived = np.mean(X_train[X_train["Survived"]==1]["Fare"])
std_fare_survived = np.std(X_train[X_train["Survived"]==1]["Fare"])
mean_fare_not_survived = np.mean(X_train[X_train["Survived"]==0]["Fare"])
std_fare_not_survived = np.std(X_train[X_train["Survived"]==0]["Fare"])

#print("mean_fare_survived = {:03.2f}".format(mean_fare_survived))
#print("std_fare_survived = {:03.2f}".format(std_fare_survived))
#print("mean_fare_not_survived = {:03.2f}".format(mean_fare_not_survived))
#print("std_fare_not_survived = {:03.2f}".format(std_fare_not_survived))

#print(gaussian(67, mean_fare_not_survived, std_fare_not_survived))
#print(gaussian(67, mean_fare_survived, std_fare_survived))

y_pred = []
for v in X_test["Fare"].values:
    if gaussian(v, mean_fare_not_survived, std_fare_not_survived) <= gaussian(v, mean_fare_survived, std_fare_survived):
        y_pred.append(1)
    else:
        y_pred.append(0)
y_pred = np.array(y_pred)
#print(y_pred)

gnb = GaussianNB()
used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("GaussianNB - Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))

gnb = MultinomialNB()
used_features =[
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("MultinomialNB - Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))