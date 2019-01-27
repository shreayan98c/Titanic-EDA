# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the Dataset
passenger_data = pd.read_csv("gender_submission.csv")
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

print("Total number of passengers:",passenger_data.shape[0])
print("Survivor Stats:\n",passenger_data.Survived.value_counts())

passenger_data.head()

train_data.head()

test_data.head()

train_data.describe(include="all")

# Check for any other unusable values NaN values
print(pd.isnull(train_data).sum())

# Plotting the Data

sns.barplot(x="Sex", y="Survived", data=train_data)

# Finding percentages of men and women who survived
print("Percentage of females who survived:", train_data["Survived"][train_data["Sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived:", train_data["Survived"][train_data["Sex"] == 'male'].value_counts(normalize = True)[1]*100)

sns.barplot(x="Pclass", y="Survived", data=train_data)

# Finding percentages of survival rate by Pclass
print("Percentage of Pclass = 1 who survived:", train_data["Survived"][train_data["Pclass"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass = 2 who survived:", train_data["Survived"][train_data["Pclass"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass = 3 who survived:", train_data["Survived"][train_data["Pclass"] == 3].value_counts(normalize = True)[1]*100)

# Taking Care of Missing Ages in the ages column
train_data["Age"] = train_data["Age"].fillna(-0.5)
test_data["Age"] = test_data["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 40, 60]
labels = ["Unknown", "Newborn", "Child", "Teenager", "Youth", "Adult", "Senior"]
train_data["AgeGroup"] = pd.cut(train_data["Age"], bins, labels = labels)
test_data["AgeGroup"] = pd.cut(test_data["Age"], bins, labels = labels)
sns.barplot(x="AgeGroup", y="Survived", data=train_data)
plt.show()