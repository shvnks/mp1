# Authors: Anthony Wong, Derek, Tyler Shanks
# Due: October 18, 2021
# COMP-472

# Importing necessary libraries
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

# https://towardsdatascience.com/implementing-a-naive-bayes-classifier-f206805a95fd?gi=f304e4496fa3
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import confusion_matrix, plot_confusion_matrix
# import matplotlib.pyplot as plt


# 3. Load the corpus using load files and make sure you set the encoding to latin1. This will read the file structure and assign the category name to each file from their parent directory name.
#files = load_files('/mp1/BBC', description='- D. Greene and P. Cunningham. \"Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering\", Proc. ICML 2006.', load_content=True, shuffle=True, encoding='latin1')
# Windows: mnt/d/Nextcloud/Concordia/6th-Semester/COMP-472/mp1/BBC
# Linux: '/home/shanks/Documents/6th-Semester/COMP-472/mp1/BBC'
cont_path = 'mp1/BBC' #simply putting /BBC or /mp1/BBC wasn't working so I used absolute path
files = load_files(cont_path, description='- D. Greene and P. Cunningham. \"Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering\", Proc. ICML 2006.', load_content=True, shuffle=True, encoding='latin1')

# print(files)

# 4. Pre-process the dataset to have the features ready to be used by a multinomial Naive Bayes classifier. This means that the frequency of each word in each class must be computed and stored in a term-document matrix. For this, you can use feature extraction.text.CountVectorizer.
X_train = files.data
y_train = files.target

X_test = files.data
y_test = files.target

# from pprint import pprint
# pprint(list(newsgroups_train.target_names))

# Feature scaling
count_vect = CountVectorizer(input=files, encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1))
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\nvect')
print(count_vect)

# X = count_vect.fit_transform(X_train)
# X_test = count_vect.transform(X_test)
#
# y = count_vect.fit_transform(y_train)
# y_test = count_vect.transform(y_test)

# X_train_count = count_vect.fit_transform(twenty_train.data)
# X_train_count.shape
# 5. Split the dataset into 80% for training and 20% for testing. For this, you must use train_test_split with the parameter random state set to None.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=None, shuffle=True, stratify=None)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# 6. Train a multinomial Naive Bayes Classifier (naive bayes.MultinomialNB) on the training set using the default parameters and evaluate it on the test set.

# Training the model
classifier = GaussianNB()
classifier.fit(X_train.toarray(), y_train)

# Predicting test results
y_pred = classifier.predict(X_test.toarray())

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 7. In a file called bbc-performance.txt, save the following information: (to make it easier for the TAs, make sure that your output for each sub-question below is clearly marked in your output file, using the headings (a), (b) . . . )


# d = '/output/bbc-performance.txt'
# with open(d, 'w+', encoding='utf-8', errors='ignore') as bbc_out:


# (a) a clear separator (a sequence of hyphens or stars) and string clearly describing the model (e.g. “Multi-nomialNB default values, try 1”)
# (b) the confusion matrix (you can use confusion matrix)
# (c) the precision, recall, and F1-measure for each class (you can use classification report)
# (d) the accuracy, macro-average F1 and weighted-average F1 of the model (you can use accuracy score and f1 score)
# (e) the prior probability of each class
# (f) the size of the vocabulary (i.e. the number of different words 1 )
# (g) the number of word-tokens in each class (i.e. the number of words in total 2 )
# (h) the number of word-tokens in the entire corpus
# (i) the number and percentage of words with a frequency of zero in each class
# (j) the number and percentage of words with a frequency of zero in the entire corpus
# (k) your 2 favorite words (that are present in the vocabulary) and their log-prob

# 8. Redo steps 6 and 7 without changing anything (do not redo step 5, the dataset split). Change the model name to something like “MultinomialNB default values, try 2” and append the results to the file bbc-performance.txt.
# 9. Redo steps 6 and 7 again, but this time, change the smoothing value to 0.0001. Append the results at the end of bbc-performance.txt.

# 10. Redo steps 6 and 7, but this time, change the smoothing value to 0.9. Append the results at the end of bbc-performance.txt.

# 11. In a separate plain text file called bbc-discussion.txt, explain in 1 to 2 paragraphs:
# (a) what metric is best suited to this dataset/task and why (see step (2))
# (b) why the performance of steps (8-10) are the same or are different than those of step (7) above.
# In total, you should have 3 output files for task 1: bbc-distribution.pdf, bbc-performance.txt, and bbc-discussion.txt.
