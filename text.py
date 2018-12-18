import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

print(twenty_train.data[0])
print(twenty_train.target_names)
print(twenty_train.target)

# представление текста в виде вектора слов
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

# формирование TF-IDF массива
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# создание объекта "три в одном" с обучением MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
_ = text_clf.fit(twenty_train.data, twenty_train.target)

# предсказание и оценка правильности
predicted = text_clf.predict(twenty_test.data)
print('MultinomialNB:',np.mean(predicted == twenty_test.target))

# создание объекта "три в одном" с обучением LogisticRegression
text_lr = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('lr', LogisticRegression(multi_class='multinomial',solver ='newton-cg'))])
_ = text_lr.fit(twenty_train.data, twenty_train.target)

predicted = text_lr.predict(twenty_test.data)
print('LogisticRegression:', np.mean(predicted == twenty_test.target))

# создание объекта "три в одном" с обучением метод опорных векторов
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)

predicted = text_clf_svm.predict(twenty_test.data)
print('SVM:',np.mean(predicted == twenty_test.target))

# создание объекта "три в одном" с обучением RandomForest
text_rf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('rf', RandomForestClassifier(n_estimators=10, criterion='entropy'))])

_ = text_rf.fit(twenty_train.data, twenty_train.target)

predicted = text_rf.predict(twenty_test.data)
print('RandomForest:',np.mean(predicted == twenty_test.target))

# гридсерч для метода опорных векторов
#parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf-svm__alpha': (1e-2, 1e-3)}
#gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
#gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)

#print(gs_clf_svm.best_score_)
#print(gs_clf_svm.best_params_)

# гридсерч для метода Naive Bayes
parameters_nb = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False)}
gs_clf = GridSearchCV(text_clf, parameters_nb, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
print(gs_clf.best_score_)
print(gs_clf.best_params_)

# а теперь с обрезанием ненужных(лишних) слов
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
print('Stemmed MultinomialNB:', np.mean(predicted_mnb_stemmed == twenty_test.target))

stemmed_count_vect = StemmedCountVectorizer(stop_words='english', ngram_range=(1,2))

text_clf_svm_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

_ = text_clf_svm_stemmed.fit(twenty_train.data, twenty_train.target)

predicted = text_clf_svm_stemmed.predict(twenty_test.data)
print('Stemmed SVM:',np.mean(predicted == twenty_test.target))