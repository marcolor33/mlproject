from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import os

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def median_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = statistics.median(y_true), statistics.median(y_pred)
    return abs((y_true - y_pred) / y_true) * 100

# Loading data
script_path = os.path.dirname(__file__)
relative_path = "data\Video_Games_Sales_as_at_22_Dec_2016.csv"
absolute_path = os.path.join(script_path, relative_path)


# Read csv
df = pd.read_csv(absolute_path)
# print(df.head())
print("\n\n\n")
# Use LabelEncoder to change category into number
# Then use OneHotEncoder(For non-string)/get_dummies(For string) to add column to the dataframe
# Test using platform column
df2 = pd.get_dummies(df, columns=['Platform', 'Genre', 'Publisher', 'Developer'])
# df2 = df
# print(df2.head())

# Drop row which has no critic score (NaN)
df3 = df2[(df2.Critic_Score.notnull())]
# print(df3.head())

df3 = shuffle(df3)
drop_column = ['Name', 'Year_of_Release', 'User_Score', 'User_Count']
df4 = df3.drop(drop_column, 1)
# print(df4.head())
# print(df4.shape)
feature_scaler = StandardScaler()
df4[['Critic_Score']] = feature_scaler.fit_transform(df4[['Critic_Score']])


# print(df4.head())

# Predict the global sales


region_list = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales']

for region in region_list:

    y_scaler = StandardScaler()
    y_scaler.fit(df4[[region]])
    df4[[region]] = y_scaler.transform(df4[[region]])

    y = df4[region]
    X = df4.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Rating'], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_poly.fit(X_train, y_train)

    predict = svr_poly.predict((X_test.values[0]).reshape(1,-1))

    print("Prediction of "+region+ " : " + str(y_scaler.inverse_transform(predict)))
    print("Actual of "+region+ " : " + str(y_scaler.inverse_transform(y_test.values[0].reshape(1,-1))))
