from __future__ import division
#
import numpy as np
import json
import pandas as pd
from IPython.display import display, HTML
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier

# classifier = LogisticRegression()

classifier = DecisionTreeRegressor()
#classifier = RandomForestClassifier()

#classifier = LinearRegression(normalize=True)


# enc = OneHotEncoder(categorical_features=[0,2,3,8,9])

def train():

    csv_path = "data.csv"


    # read data from csv
    df = pd.read_csv(csv_path)


    # remove row that have no value
    df = df.dropna()
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.loc[(df['Publisher'] == 'Nintendo') & (df['Platform'] == 'Wii')]
    print(df.shape)


    y_train = df['Global_Sales'].values[:40]
    y_test = df['Global_Sales'].values[40:]


    # remove name and sales
    df = df.drop(columns=['Name', 'NA_Sales', 'EU_Sales','JP_Sales','Other_Sales','Global_Sales', 'Year_of_Release', 'Rating', 'Publisher', 'Developer', 'Platform'])



    # method 2
    # transform the string feature into many binary features
    cols_to_transform = [ 'Genre']
    expanded_df = pd.get_dummies(df, columns=cols_to_transform , prefix=cols_to_transform)
    print(expanded_df)


    x_train = expanded_df.values[:40]
    x_test = expanded_df.values[40:]


    #df1 = df[['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']]
    #x_train = df.values[:6000]
    #x_test = df.values[6000:]

    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)

    print(prediction)

    print("-----------------")

    print(y_test)

    print("-----------------")

    error = np.true_divide(np.absolute(prediction - y_test),y_test) * 100
    print(error)
    print("average error %: ", error.sum()/error.shape)
    print("max. error in %: ", np.amax(error))






train()
