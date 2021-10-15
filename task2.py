import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# TASK 2 PART 2: Load the Drug.csv in Python by using panda.read_csv.
df = pd.read_csv("drug200.csv")
# print(df)

# TASK 2 PART 3: Plot the distribution of the instances in each class and store the graphic in a file called drug-distribution.pdf.
df.groupby("Drug")["Na_to_K"].nunique().plot(kind="bar")
plt.savefig("drug-distribution.pdf")

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
train_test_split(df)
