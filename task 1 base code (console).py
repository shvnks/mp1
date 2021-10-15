# Authors: Anthony Wong, Derek Lam, Tyler Shanks
# Due: October 18, 2021
# COMP-472

# Importing necessary libraries
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
# task 1 step 3
files = load_files('mp1/BBC', encoding='latin1')

# task 1 step 4
labels, counts = np.unique(files.target, return_counts = True)
labels_str = np.array(files.target_names)[labels]
print(dict(zip(labels_str, counts)))    # creating dictionary

# task 1 step 5 splitting dataset
X_train, X_test, y_train, y_test = train_test_split(files.data, files.target, train_size=0.8, test_size=0.2)
print(len(X_train), len(X_test))

vectorizer = TfidfVectorizer(stop_words="english", max_features=2000) #max_features can be changed
vectorizer.fit(X_train)

# task 1 step 6
bayes = MultinomialNB()
bayes.fit(vectorizer.transform(X_train), y_train)
y_predict = bayes.predict(vectorizer.transform(X_test))
print(accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))
