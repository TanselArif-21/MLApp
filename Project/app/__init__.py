from flask import Flask, request, redirect, url_for, flash, jsonify, render_template, send_from_directory
from functions import return_something, return_something2, get_prediction_lr
import flask_excel as excel
import pandas as pd
import os
import string
import random

# This line sets the app directory as the working directory
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# The home route
@app.route('/', methods=['GET'])
def home_page():
    # Show the index page
    return render_template('index.html')


# The dosomething route
@app.route('/dosomething', methods=['GET'])
def dosomething():
    # Show the index page
    return str(return_something())


# The dosomething2 route
@app.route('/dosomething2', methods=['GET', 'POST'])
def dosomething2():
    # If the request is a post request
    if request.method == 'POST':
        # This line creates a dataframe from a list of lists
        df = pd.DataFrame(request.get_array(field_name='file'))

        # The first row is the list of column names so set the column names to the first row
        df.columns = df.iloc[0, :]

        # Now remove the first row
        df = df[1:]

        # Print the dataframe to the console
        print(df)

        # Print the column names to the console
        print(df.columns)

        # These are the values given for the fields
        input_list = list(request.form.values())

        filename = ''
        for i in range(4):
            filename = filename + random.choice(string.ascii_letters)

        # These are the names of the fields, i.e. LotFrontage, LotArea etc...
        input_names = list(request.form)

        result = get_prediction_lr(df, input_list, filename)

        # This code section creates a dictionary where each key is a row in the dataframe
        res = dict()
        for i in range(len(df)):
            res[i] = list(df.iloc[i, :])

        # Return the result
        #return 'The predicted Sale Price of this house is: ' + str(round(result, 2)) + '. \n <button><a href="/plot/' + filename + '">See Sales Price Distribution</a></button><p></p>'
        return 'The predicted Sale Price of this house is: ' + str(round(result, 2)) + '. <p></p> <img src="static/' + filename + '.png"><p></p>'
    else:
        # Show the form page
        return render_template('dosomethingform.html')


# A route to the test page that simply returns hello
@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    # Initiate the excel part of flask
    excel.init_excel(app)
    # Let the console know that the load is successful
    print("loaded OK")

    # Set to debug mode
    app.run(debug=True)
