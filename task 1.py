# Authors: Anthony Wong, Derek Lam, Tyler Shanks
# Due: October 18, 2021
# COMP-472

# Importing necessary libraries
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, plot_confusion_matrix

# Step 3
files = load_files('mp1/BBC', encoding='latin1')

# Step 4
labels, counts = np.unique(files.target, return_counts=True)
labels_str = np.array(files.target_names)[labels]
print(dict(zip(labels_str, counts)))    # creating dictionary

# Step 5
X_train, X_test, y_train, y_test = train_test_split(files.data, files.target, train_size=0.8, test_size=0.2)
print("training set:", len(X_train), "  test set:", len(X_test))

vectorizer = TfidfVectorizer(stop_words="english", decode_error="ignore", max_features=2500)  # max_features can be changed
vectorizer.fit(X_train)

tfidf_matrix = vectorizer.fit_transform(X_train)
feature_names = vectorizer.get_feature_names_out()

print(tfidf_matrix.shape)

# Step 6
multiBayes1 = MultinomialNB()
multiBayes1.fit(vectorizer.transform(X_train), y_train)
y_predict = multiBayes1.predict(vectorizer.transform(X_test))

# Step 7
print("\n********** MultinomialNB default values, try 1 **********")
print("b) Confusion Matrix: \n", confusion_matrix(y_test, y_predict))
print("\nc) Classification Report: \n", classification_report(y_test, y_predict, target_names=labels_str))
print("d) Accuracy:", accuracy_score(y_test, y_predict))
print("   Macro F1:", f1_score(y_test, y_predict, average='macro'))
print("   Weighted F1:", f1_score(y_test, y_predict, average='weighted'))
print("e) Prior probabilities:", multiBayes1.class_log_prior_)  # not sure
print("f) Vocabulary Size:", len(vectorizer.vocabulary_))
print("g) Number of word-tokens in each class:", multiBayes1.class_count_)
print("h) Number of word-tokens in entire corpus:", tfidf_matrix.sum())
print("i) Number and percentage of words with zero  freq in each class")
print("j) Number and percentage of word with one freq in entire corpus")
print("k) ")  # temp number
