import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""Data Dictionary
Variable :	Definition	Key
survival :	Survival	0 = No, 1 = Yes
pclass :	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex :	Sex	
Age :	Age in years	
sibsp :	# of siblings / spouses aboard the Titanic	
parch :	# of parents / children aboard the Titanic	
ticket :	Ticket number	
fare :	Passenger fare	
cabin :	Cabin number	
embarked :	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
"""
df = pd.read_csv("train.csv")
col_2_drop = ["Fare", "Cabin", "Ticket", "Embarked"]
df = df.drop(columns=col_2_drop)
df.info()

# visualization SibSp
fig, ax = plt.subplots(1, 2)
sns.countplot(x="SibSp", hue="Survived", data=df, palette="Set2", ax=ax[0])
for con in ax[0].containers:
    ax[0].bar_label(con, label_type="edge", fontsize=10, padding=3)
sns.kdeplot(x="SibSp", hue="Survived", data=df, ax=ax[1])
plt.show()

""" from the above visualization, we can see that most of the passengers had 0 or 1 siblings/spouses aboard the Titanic, 
and very few had more than 2. The survival rate is higher for those with fewer siblings/spouses.
"""

# create SibSp Bracket
sibsp_bins = [0, 1, 2, np.inf]
sibsp_label = ["0", "1", "2+"]

df["Sib_Bracket"] = pd.cut(
    df["SibSp"].astype("int32"),
    bins=sibsp_bins,
    labels=sibsp_label,
    include_lowest=True,
    right=False,
)

df.drop(columns=["SibSp", "Name"], inplace=True)


# visualization Pclass
sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set2")
plt.show()


# visualization Age column
fig, ax = plt.subplots(1, 2)
sns.histplot(x="Age", data=df, palette="Set2", ax=ax[0])
sns.boxplot(x="Age", data=df, palette="Set2", ax=ax[1])
plt.show()
"""__Summary of Age column__
The Age column has a outliers, as seen in the boxplot.
So, i gonna fill the missing values with the median of the Age column.
The median is a better measure of central tendency for this column, as it is less affected by outliers.
"""
df["Age"].fillna(df["Age"].median(), inplace=True)

df.sample(10)


df_done = pd.get_dummies(df, dtype="int32")
df_done.info()

"""
    //TODO :
"""
