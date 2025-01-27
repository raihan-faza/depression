import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# importing dataset
df = pd.read_csv("data.csv")
print(df.info())

# checking null values
print(df.isna().sum())

# looks like there is a nan value in financial stress, well its only 3 maybe we should delete that
df.dropna(inplace=True)
print(df.isna().sum())

# changin object type data to numerical data
encoder = LabelEncoder()
object_columns = df.select_dtypes(include=["object"]).columns
print(df[object_columns].info())
for column in object_columns:
    df[column + "_encoded"] = encoder.fit_transform(df[column])
    df.drop(columns=column, inplace=True)
print(df.info())
# eda tipis - tipis
# who depressed more? men or women?
# men udah percaya aja, jk
"""
sns.countplot(data=df, x="Gender")
plt.title("Count Plot of Categories")
plt.xlabel("Gender")
plt.ylabel("Total")
plt.show()
"""

# how does each data correlates
sns.heatmap(df.corr(), annot=True, cmap="crest")
plt.show()
