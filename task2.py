import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


# TASK 2 PART 2: Load the Drug.csv in Python by using panda.read_csv.
df = pd.read_csv("drug200.csv")

# TASK 2 PART 3: Plot the distribution of the instances in each class and store the graphic in a file called drug-distribution.pdf.
df.groupby("Drug")["Na_to_K"].nunique().plot(kind="bar")
plt.savefig("output/task2/drug-distribution.pdf")

# TASK 2 PART 4: Convert all ordinal and nominal features in numerical format

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

# TASK 2 PART 5: Split the dataset using train_test_split using the default parameter values.
X = df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]] # features
y = df["Drug"]                                         # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y)

# TASK 2 PART 6a: NB: a Gaussian Naive Bayes Classier (naive bayes.GaussianNB) with the default parameters.
# Console indicator for dataset
print('####################################\n########     GaussianNB     ########\n####################################')
gauss = GaussianNB()                        # Create GaussianNB object
gauss.fit(X_train, y_train)                 # Train GaussianNB
y_predict = gauss.predict(X_test)           # Predict response from the test dataset

# Making the confusion matrix
print("b) Confusion Matrix: ")
print(confusion_matrix(y_test, y_predict))
print()

# Print output
print("c & d) Classification Report: \n", metrics.classification_report(y_test, y_predict))

# TASK 2 PART 6b: Base-DT: a Decision Tree (tree.DecisionTreeClassifier) with the default parameters.
# Console indicator for dataset
print('\n####################################\n#####  DecisionTreeClassifier  #####\n####################################')
dtClass = DecisionTreeClassifier()          # Create DecisionTreeClassifier object
dtClass = dtClass.fit(X_train, y_train)     # Train DecisionTreeClassifier
y_predict = dtClass.predict(X_test)         # Predict response from the test dataset

# Making the confusion matrix
print("b) Confusion Matrix: ")
print(confusion_matrix(y_test, y_predict))
print()

# Print output
print("c & d) Classification Report: \n", metrics.classification_report(y_test, y_predict))

# TASK 2 PART 6c: Top-DT: a better performing Decision Tree found using (GridSearchCV). The gridsearch will allow you tond the best combination of hyper-parameters, as determined by the evaluation function that you have determined in step (3) above. The hyper-parameters that you will experiment with are:
# • criterion: gini or entropy
# • max_depth : 2 different values of your choice
# • min_samples_split: 3 different values of your choice
# Console indicator for dataset
print('\n####################################\n#######  DT w/ GridSearchCV  #######\n####################################')
# dtGrid = DecisionTreeClassifier()          # Create DecisionTreeClassifier object
# dtGrid = dtGrid.fit(X_train, y_train)     # Train DecisionTreeClassifier
# y_predict = dtGrid.predict(X_test)         # Predict response from the test dataset

parameters = {'criterion':['gini', 'entropy'], 'max_depth':[1,3], 'min_samples_split':[5,10,15]}

grid = GridSearchCV(dtClass, param_grid=parameters, cv=10, n_jobs=-1)

grid.fit(X_train, y_train)
print('Grid best parameters: ', grid.best_estimator_)
print('Grid best score: ', grid.best_score_)

# TASK 2 PART 6d: PER: a Perceptron (linear model.Perceptron), with default parameter values.
# Console indicator for dataset
print('\n####################################\n########     Perceptron     ########\n####################################')
per = Perceptron()          # Create Perceptron object
per = per.fit(X_train, y_train)     # Train Perceptron
y_predict = per.predict(X_test)         # Predict response from the test dataset

# Making the confusion matrix
print("b) Confusion Matrix: ")
print(confusion_matrix(y_test, y_predict))
print()

# Print output
print("c & d) Classification Report: \n", metrics.classification_report(y_test, y_predict, zero_division=0))

# TASK 2 PART 6e: Base-MLP: a Multi-Layered Perceptron (neural network.MLPClassifier) with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.
# Console indicator for dataset
print('\n####################################\n##### Multi-Layered Perceptron #####\n####################################')
<<<<<<< HEAD
mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=5000, activation='logistic', solver='sgd')          # Create MLPClassifier object
=======
mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=4000, activation='logistic', solver='sgd')          # Create MLPClassifier object
>>>>>>> 416ca97ec1d80cf9e5195d3986df00eef5b7e5ed
mlp = mlp.fit(X_train, y_train)     # Train MLPClassifier
y_predict = mlp.predict(X_test)         # Predict response from the test dataset

# Making the confusion matrix
print("b) Confusion Matrix: ")
print(confusion_matrix(y_test, y_predict))
print()

# Print output
print("c & d) Classification Report: \n", metrics.classification_report(y_test, y_predict, zero_division=0))


# TASK 2 PART 6f: Top-MLP: a better performing Multi-Layered Perceptron found using grid search. For this, you need to experiment with the following parameter values:
# • activation function: sigmoid, tanh, relu and identity
# • 2 network architectures of your choice: for eg 2 hidden layers with 30+50 nodes, 3 hidden layers with 10 + 10 + 10
# • solver: Adam and stochastic gradient descent
# Console indicator for dataset
print('\n####################################\n######   MLP w/ GridSearchCV  ######\n####################################')

parameters = {'hidden_layer_sizes':[(30,50), (10,10,10)], 'activation':['logistic', 'tanh', 'relu', 'identity'], 'solver':['sgd', 'adam']}

grid = GridSearchCV(mlp, param_grid=parameters, cv=10, n_jobs=-1)

grid.fit(X_train, y_train)
print('Grid best parameters: ', grid.best_estimator_)
print('Grid best score: ', grid.best_score_)

# TASK 2 PART 7: For each of the 6 classfier above, append the following information in ale called drugs-performance.txt:(to make it easier for the TAs, make sure that your output for each sub-question below is clearly marked in your output le, using the headings (a), (b) . . . )


# TASK 2 PART 7a: a clear separator (a sequence of hyphens or stars) and a string clearly describing the model (e.g. the model name + hyper-parameter values that you changed). In the case of Top-DT and Top-MLP, display the best hyperparameters found by the gridsearch.
