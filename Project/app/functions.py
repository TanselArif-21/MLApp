import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def return_something():
    return 200


def return_something2(s):
    return s


def plot_df(cols, filename):
    x = np.random.randn(100)
    ax = sns.distplot(list(cols))
    plt.savefig('static/' + filename)
    return


def get_prediction_lr(df, ls, filename):
    '''
    Description: This function accepts a dataframe of house prices and a user provided list to predict the sales price for
    df: a data frame
    ls: a list containing the information for a house to predict for
    '''

    # Save a plot image
    plot_df(df['SalePrice'], filename)

    # This is a linear regression object
    linreg = LinearRegression()

    # This is our predictors dataframe
    x = df.drop('SalePrice', axis=1)

    # We fit the model using the predictors
    linreg.fit(x, df['SalePrice'])

    # This is our observation we would like to predict. Initially we use the means of all observations
    # and we overwrite the fields which are provided by the user. i.e. the default of a field
    # is the mean for that field
    pred_x = x.mean().values.reshape(1, -1)

    # Loop through the user provided list of fields and update the list for our observation
    # whenever a field is supplied
    for i in range(len(ls)):
        if ls[i] != '':
            pred_x[0][i] = float(ls[i])

    # Return the predicted Sale Price
    return list(linreg.predict(pred_x))[0]
