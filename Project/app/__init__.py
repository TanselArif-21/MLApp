from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
from functions import return_something

# This line sets the app directory as the working directory
app = Flask(__name__)


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


# A route to the test page that simply returns hello
@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'


if __name__ == '__main__':

    # Let the console know that the load is successful
    print("loaded OK")

    # Set to debug mode
    app.run(debug=True)
