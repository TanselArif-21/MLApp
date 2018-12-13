from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
from functions import return_something,return_something2
import flask_excel as excel

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


# The dosomething2 route
@app.route('/dosomething2', methods=['GET', 'POST'])
def dosomething2():
    # If the request is a post request
    if request.method == 'POST':
        return jsonify({"result": request.get_array(field_name='file')})
    else:
        # Show them a form
        return '''
        <!doctype html>
        <title>Upload an excel file</title>
        <h1>Excel file upload (csv)</h1>
        <form action="" method=post enctype=multipart/form-data><p>
        <input type=file name=file><input type=submit value=Upload>
        </form>
        '''


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
