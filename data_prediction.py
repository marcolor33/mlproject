from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import os

# Loading data
script_path = os.path.dirname(__file__)
relative_path = ".data\Video_Games_Sales_as_at_22_Dec_2016.csv"
absolute_path = os.path.join(script_path, relative_path)

# Read csv
df = pd.read_csv(absolute_path)
print(df.head())

# Use LabelEncoder to change category into number
# Then use OneHotEncoder(For non-string)/get_dummies(For string) to add column to the dataframe
# Test using platform column
df2 = pd.get_dummies(df, columns=['Platform'])
print(df2.head())
