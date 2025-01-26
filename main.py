import pandas as pd
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
print(df.info())

# now what should i do?
