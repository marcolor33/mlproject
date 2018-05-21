from __future__ import division
#
import numpy as np
import json
import pandas as pd
from IPython.display import display, HTML
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

# classifier = LogisticRegression()

classifier = DecisionTreeRegressor()

# enc = OneHotEncoder(categorical_features=[0,2,3,8,9])

def train():

    csv_path = "data.csv"


    # read data from csv
    df = pd.read_csv(csv_path)


    # remove row that have no value
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)


    print(df.shape)


    y_train = df['Global_Sales'].values[:6000]
    y_test = df['Global_Sales'].values[6000:]


    # remove name and sales
    df = df.drop(columns=['Name', 'NA_Sales', 'EU_Sales','JP_Sales','Other_Sales','Global_Sales'])



    # method 2
    # transform the string feature into many binary features
    cols_to_transform = [ 'Platform', 'Genre', 'Publisher', 'Developer','Rating']
    expanded_df = pd.get_dummies(df, columns=cols_to_transform , prefix=cols_to_transform)


    x_train = expanded_df.values[:6000]
    x_test = expanded_df.values[6000:]

    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)

    print(prediction)

    print("-----------------")

    print(y_test)

    print("-----------------")

    error = np.true_divide(np.absolute(prediction - y_test),y_test) * 100
    print(error)






train()
