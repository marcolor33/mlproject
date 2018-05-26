from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.svm import SVC
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
print("\n\n\n")

df = df[(df.Genre.notnull())]

le = LabelEncoder()
le.fit(df['Genre'])
df['Genre'] = le.transform(df['Genre'])
print(le.inverse_transform(df['Genre']))

df = pd.get_dummies(df, columns=['Platform', 'Publisher', 'Developer', 'Rating'])
print(df.head())

# Drop row which has no critic score (NaN)
df = df[(df.Critic_Score.notnull())]
print(df.head())
print(df.shape)

df = shuffle(df)
drop_column = ['Name', 'Year_of_Release', 'User_Score', 'User_Count']
df = df.drop(drop_column, 1)
print(df.head())
print(df.shape)

# Standardization
scaler = StandardScaler()
df[['Critic_Score', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']] = scaler.fit_transform(df[['Critic_Score', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']])
print(df.head())
print(df.shape)

# Predict the genre
y = df['Genre']
X = df.drop(['Genre'], 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# Using SVC
# Very Time-Consuming (OneVsRestClassifier)
classif = OneVsRestClassifier(SVC())
classif.fit(X_train, y_train)
y_predict = classif.predict(X_test)
print(accuracy_score(y_test, y_predict))
print(f1_score(y_test, y_predict, average=None))
print(f1_score(y_test, y_predict, average='micro'))
print(f1_score(y_test, y_predict, average='macro'))
