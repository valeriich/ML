from sklearn import datasets
import matplotlib.pyplot as plt

# Loading dataset
iris_df = datasets.load_iris()

# Available methods on dataset
print(dir(iris_df))

# Features
print(iris_df.feature_names)

# Targets
print(iris_df.target)

# Target Names
print(iris_df.target_names)

label = {0: 'red', 1: 'blue', 2: 'green'}

# Dataset Slicing
x_axis = iris_df.data[:, 0]  # Sepal Length
y_axis = iris_df.data[:, 2]  # Sepal Width ??? petal length (cm)!!


# Plotting
plt.scatter(x_axis, y_axis, c=iris_df.target)
plt.show()

from sklearn.cluster import KMeans

# Declaring Model
model = KMeans(n_clusters=3)

# Fitting Model
model.fit(iris_df.data)

# Predicitng a single input
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])
print(predicted_label)

# Prediction on the entire data
all_predictions = model.predict(iris_df.data)

# Printing Predictions
print(all_predictions)

# number of group is set randomly by K-mean model!

all_predictions[all_predictions==1] = 7
all_predictions[all_predictions==0] = 1
all_predictions[all_predictions==7] = 0
print(all_predictions)
print(sum(all_predictions==iris_df.target)/len(iris_df.target))
