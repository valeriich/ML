from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import pickle

# ЭТАП 2
# Обучение моделей
args = {
  #path to serialized db of facial embeddings
  "embeddings": "face/output/embeddings.pickle",
  #path to output model trained to recognize faces
  "recognizer": "face/output/recognizer.pickle",
  #ath to output label encoder
  "le": "face/output/le.pickle"
}

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")

# метод опорных векторов
#recognizer = SVC(C=1.0, kernel="linear", probability=True)

# логистическая регрессия
# recognizer = LogisticRegression()

# Naive Bayes
#recognizer = BernoulliNB()
#recognizer.fit(data["embeddings"], labels)

# KNN
from sklearn.neighbors import KNeighborsClassifier

recognizer = KNeighborsClassifier(n_neighbors=3)
recognizer.fit(data["embeddings"], labels)
res = recognizer.predict([data["embeddings"][11]])
print(res)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
