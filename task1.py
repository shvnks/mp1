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
from matplotlib import pyplot as plt
import os

# Step 2
# 510 files
businessList = os.listdir('BBC/business')
business = len(businessList)
# 386 files
entertainmentList = os.listdir('BBC/entertainment')
entertainment = len(entertainmentList)
# 417 files
politicList = os.listdir('BBC/politics')
politics = len(politicList)
# 511 files
sportList = os.listdir('BBC/sport')
sport = len(sportList)
# 401 files
techList = os.listdir('BBC/tech')
tech = len(techList)

# plotting chart
dev_x = ['business', 'entertainment', 'politics', 'sport', 'tech']
dev_y = [business, entertainment, politics, sport, tech]

plt.bar(dev_x, dev_y)
plt.title("Documents per section")
plt.ylabel("Number of documents")
plt.tight_layout()

plt.savefig("output/task1/bbc-distribution.pdf", bbox_inches="tight")

# Step 3
files = load_files('BBC', encoding='latin1')

# Step 4
labels, counts = np.unique(files.target, return_counts=True)
labels_str = np.array(files.target_names)[labels]
print(dict(zip(labels_str, counts)))

# Step 5
X_train, X_test, y_train, y_test = train_test_split(files.data, files.target, train_size=0.8, test_size=0.2)
print("training set:", len(X_train), "  test set:", len(X_test))

vectorizer = TfidfVectorizer(stop_words="english", decode_error="ignore")  # max_features can be changed
vectorizer.fit(X_train)

tfidf_matrix = vectorizer.transform(X_train)
feature_names = vectorizer.get_feature_names_out()

print(tfidf_matrix.shape)

# Step 6
multiBayes = MultinomialNB()
multiBayes.fit(vectorizer.transform(X_train), y_train)
y_predict = multiBayes.predict(vectorizer.transform(X_test))

# for g), i) and j)
word_per_class = [0, 0, 0, 0, 0]
zeroFreq = [0, 0, 0, 0, 0]
word_per_class = [0, 0, 0, 0, 0]
oneFreq = 0
for category in range(0, len(files.target_names)):
    feature = multiBayes.feature_count_[category]
    for count in feature:
        word_per_class[category] += count
        if count == 0:
            zeroFreq[category] += 1
        if count == 1:
            oneFreq += 1

# for k)
my_words = ['Christmas', 'tree']
my_word_matrix = vectorizer.transform(my_words)

# Step 7
fh = open('bbc-performance.txt', 'w')
fh.write("\n********** MultinomialNB default values, try 1 **********\n")
fh.write(str(dict(zip(labels_str, counts))))
fh.write("\nb) Confusion Matrix:\n" + str(confusion_matrix(y_test, y_predict)))
fh.write("\n\nc) Classification Report: \n" + str(classification_report(y_test, y_predict, target_names=labels_str)))
fh.write("\nd) Accuracy:" + str(accuracy_score(y_test, y_predict)))
fh.write("\n   Macro F1:" + str(f1_score(y_test, y_predict, average='macro')))
fh.write("\n   Weighted F1:" + str(f1_score(y_test, y_predict, average='weighted')))
fh.write("\ne) Prior probabilities:" + str(multiBayes.class_log_prior_))
fh.write("\nf) Vocabulary Size:" + str(len(vectorizer.vocabulary_)))
fh.write("\ng) Number of word-tokens in each class:" + str(word_per_class))  # multiBayes.class_count_
fh.write("\nh) Number of word-tokens in entire corpus:" + str(tfidf_matrix.sum()))
fh.write("\ni) Number and percentage of words with zero frequency in each class:" + str(zeroFreq))
fh.write("\nj) Number and percentage of word with one frequency in entire corpus:" + str(oneFreq))
fh.write("\nk) Favorite words: Christmas and tree - " + str(multiBayes.predict_log_proba(my_word_matrix)))


# Step 8 ------------------------------------------------------------------------------------------------------------
multiBayes = MultinomialNB()
multiBayes.fit(vectorizer.transform(X_train), y_train)
y_predict = multiBayes.predict(vectorizer.transform(X_test))

word_per_class = {}
zeroFreq = [0, 0, 0, 0, 0]
word_per_class = [0, 0, 0, 0, 0]
oneFreq = 0
for category in range(0, len(files.target_names)):
    feature = multiBayes.feature_count_[category]
    for count in feature:
        word_per_class[category] += count
        if count == 0:
            zeroFreq[category] += 1
        if count == 1:
            oneFreq += 1

fh.write("\n\n********** MultinomialNB default values, try 2 **********\n")
fh.write(str(dict(zip(labels_str, counts))))
fh.write("\nb) Confusion Matrix:\n" + str(confusion_matrix(y_test, y_predict)))
fh.write("\n\nc) Classification Report: \n" + str(classification_report(y_test, y_predict, target_names=labels_str)))
fh.write("\nd) Accuracy:" + str(accuracy_score(y_test, y_predict)))
fh.write("\n   Macro F1:" + str(f1_score(y_test, y_predict, average='macro')))
fh.write("\n   Weighted F1:" + str(f1_score(y_test, y_predict, average='weighted')))
fh.write("\ne) Prior probabilities:" + str(multiBayes.class_log_prior_))  # not sure
fh.write("\nf) Vocabulary Size:" + str(len(vectorizer.vocabulary_)))
fh.write("\ng) Number of word-tokens in each class:" + str(word_per_class))  # multiBayes.class_count_
fh.write("\nh) Number of word-tokens in entire corpus:" + str(tfidf_matrix.sum()))
fh.write("\ni) Number and percentage of words with zero frequency in each class:" + str(zeroFreq))
fh.write("\nj) Number and percentage of word with one frequency in entire corpus:" + str(oneFreq))
fh.write("\nk) Favorite words: Christmas and tree - " + str(multiBayes.predict_log_proba(my_word_matrix)))


# Step 9 ------------------------------------------------------------------------------------------------------------
multiBayes = MultinomialNB(alpha=0.0001)
multiBayes.fit(vectorizer.transform(X_train), y_train)
y_predict = multiBayes.predict(vectorizer.transform(X_test))

word_per_class = {}
zeroFreq = [0, 0, 0, 0, 0]
word_per_class = [0, 0, 0, 0, 0]
oneFreq = 0
for category in range(0, len(files.target_names)):
    feature = multiBayes.feature_count_[category]
    for count in feature:
        word_per_class[category] += count
        if count == 0:
            zeroFreq[category] += 1
        if count == 1:
            oneFreq += 1

fh.write("\n\n********** MultinomialNB smoothing 0.0001 **********\n")
fh.write(str(dict(zip(labels_str, counts))))
fh.write("\nb) Confusion Matrix:\n" + str(confusion_matrix(y_test, y_predict)))
fh.write("\n\nc) Classification Report: \n" + str(classification_report(y_test, y_predict, target_names=labels_str)))
fh.write("\nd) Accuracy:" + str(accuracy_score(y_test, y_predict)))
fh.write("\n   Macro F1:" + str(f1_score(y_test, y_predict, average='macro')))
fh.write("\n   Weighted F1:" + str(f1_score(y_test, y_predict, average='weighted')))
fh.write("\ne) Prior probabilities:" + str(multiBayes.class_log_prior_))  # not sure
fh.write("\nf) Vocabulary Size:" + str(len(vectorizer.vocabulary_)))
fh.write("\ng) Number of word-tokens in each class:" + str(word_per_class))  # multiBayes.class_count_
fh.write("\nh) Number of word-tokens in entire corpus:" + str(tfidf_matrix.sum()))
fh.write("\ni) Number and percentage of words with zero frequency in each class:" + str(zeroFreq))
fh.write("\nj) Number and percentage of word with one frequency in entire corpus:" + str(oneFreq))
fh.write("\nk) Favorite words: Christmas and tree - " + str(multiBayes.predict_log_proba(my_word_matrix)))


# Step 10 -----------------------------------------------------------------------------------------------------------
multiBayes = MultinomialNB(alpha=0.9)
multiBayes.fit(vectorizer.transform(X_train), y_train)
y_predict = multiBayes.predict(vectorizer.transform(X_test))

word_per_class = {}
zeroFreq = [0, 0, 0, 0, 0]
word_per_class = [0, 0, 0, 0, 0]
oneFreq = 0
for category in range(0, len(files.target_names)):
    feature = multiBayes.feature_count_[category]
    for count in feature:
        word_per_class[category] += count
        if count == 0:
            zeroFreq[category] += 1
        if count == 1:
            oneFreq += 1

fh.write("\n\n********** MultinomialNB smoothing 0.9 **********\n")
fh.write(str(dict(zip(labels_str, counts))))
fh.write("\nb) Confusion Matrix:\n" + str(confusion_matrix(y_test, y_predict)))
fh.write("\n\nc) Classification Report: \n" + str(classification_report(y_test, y_predict, target_names=labels_str)))
fh.write("\nd) Accuracy:" + str(accuracy_score(y_test, y_predict)))
fh.write("\n   Macro F1:" + str(f1_score(y_test, y_predict, average='macro')))
fh.write("\n   Weighted F1:" + str(f1_score(y_test, y_predict, average='weighted')))
fh.write("\ne) Prior probabilities:" + str(multiBayes.class_log_prior_))  # not sure
fh.write("\nf) Vocabulary Size:" + str(len(vectorizer.vocabulary_)))
fh.write("\ng) Number of word-tokens in each class:" + str(word_per_class))  # multiBayes.class_count_
fh.write("\nh) Number of word-tokens in entire corpus:" + str(tfidf_matrix.sum()))
fh.write("\ni) Number and percentage of words with zero frequency in each class:" + str(zeroFreq))
fh.write("\nj) Number and percentage of word with one frequency in entire corpus:" + str(oneFreq))
fh.write("\nk) Favorite words: Christmas and tree - " + str(multiBayes.predict_log_proba(my_word_matrix)))

fh.close()
