import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification

# TASK 2 PART 2: Load the Drug.csv in Python by using panda.read_csv.
df = pd.read_csv("drug200.csv")
# print(df)

# TASK 2 PART 3: Plot the distribution of the instances in each class and store the graphic in a file called drug-distribution.pdf.
df.groupby("Drug")["Na_to_K"].nunique().plot(kind="bar")
plt.savefig("output/task2/drug-distribution.pdf")

# TASK 2 PART 4: Convert all ordinal and nominal features in numerical format
# sexe = pd.get_dummies(df, columns=["Sex"], drop_first=True)
# print(sexe)
# print(pd.Categorical(sexe))

# BP = pd.get_dummies(df, columns=["BP"], drop_first=True)
# print(BP)

# cholesterol = pd.get_dummies(df["Cholesterol"])
# cholesterol = pd.get_dummies(df, columns=["Cholesterol"], drop_first=True)

# Converting categorie Sex
df.Sex = pd.Categorical(df.Sex, ["F", "M"])
df.Sex = df.Sex.cat.codes

# Converting categorie Cholesterol
df.Cholesterol = pd.Categorical(df.Cholesterol, ["HIGH", "NORMAL"])
df.Cholesterol = df.Cholesterol.cat.codes

# Converting categorie BP
df.BP = pd.Categorical(df.BP, ["HIGH", "NORMAL", "LOW"])
df.BP = df.BP.cat.codes

# Converting categorie Drug
df.Drug = pd.Categorical(df.Drug, ["drugA", "drugB", "drugC", "drugX", "drugY"])
df.Drug = df.Drug.cat.codes

# print(df)

# TASK 2 PART 5: Split the dataset using train_test_split using the default parameter values.
# print(train_test_split(df))
# X_train, X_test = train_test_split(df)
X_train, X_test, y_train, y_test = train_test_split(X, y)
#
print(X_train)
print(X_test)
print(y_train)
print(y_test)
# TASK 2 PART 6a: NB: a Gaussian Naive Bayes Classier (naive bayes.GaussianNB) with the default parameters.
gauss = GaussianNB()
gauss.fit(X_train)
print(gauss)

# TASK 2 PART 6b:

# TASK 2 PART 6c:

# TASK 2 PART 6d:

# TASK 2 PART 6e:

# TASK 2 PART 6f:

# TASK 2 PART 7: For each of the 6 classier above, append the following information in ale called drugs-performance.txt:(to make it easier for the TAs, make sure that your output for each sub-question below is clearly marked in your output le, using the headings (a), (b) . . . )

# TASK 2 PART 7a: a clear separator (a sequence of hyphens or stars) and a string clearly describing the model (e.g. the model name + hyper-parameter values that you changed). In the case of Top-DT and Top-MLP, display the best hyperparameters found by the gridsearch.
