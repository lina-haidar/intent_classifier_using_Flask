# -*- coding: utf-8 -*-

import os
import argparse
from flask import Flask, render_template, jsonify, request, Response, redirect, url_for
from intent_classifier import IntentClassifier


app = Flask(__name__)



classes = ['abbreviation', 'aircraft', 'aircraft+flight+flight_no', 'airfare',
       'airfare+flight', 'airfare+flight_time', 'airline', 'airline+flight_no',
       'airport', 'capacity', 'cheapest', 'city', 'day_name', 'distance',
       'flight', 'flight+airfare', 'flight+airline', 'flight_no',
       'flight_no+airline', 'flight_time', 'ground_fare', 'ground_service',
       'ground_service+ground_fare', 'meal', 'quantity', 'restriction']

model = IntentClassifier( classes)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/ready',  methods=['GET', 'POST'])
def ready():
    """
    ready() renders the template intent.html when the server is running, model has been loaded and is ready to serve infer requests and it returns response body "Not ready" when the model has not been loaded.
    """
    if request.method =="GET": 
        if model.is_ready():
        #return 'OK', 200
            return render_template('intent.html', status = "Status: OK")
            #return redirect(url_for("intent"))
        else:
        #return 'Not ready', 423
            return render_template('home.html', status =  " Status: Not ready")


@app.route('/intent',  methods= ['GET', 'POST'])  
def intent():
    """
    intent() takes the user input as a string then it checks: 
    1- whether the string is empty
    2- whether the string is human readable
    
    If the string is not empty and is human readable, then the function returns the output of _model.predict_ to the user interface.
    
    """

    if request.method =="POST":
        # Take the user input as a text
        text = request.form["query"]
        
        # Check if the text is empty
        if text.strip():
            
            # Check if the text is human readable (using PyDictionary)
            if model.check_text(text):
                prediction = model.predict(text)
        
                return render_template('intent.html', output= prediction)
        
            else:
                error = text + " is not a valid text"
                return render_template('intent.html', output= error)
        else:
            error = "Error: Text is missing"
            return render_template('intent.html', output= error)
        
            



def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, required=True, help='Path to model directory or file.')
    arg_parser.add_argument('--port', type=int, default=os.getenv('PORT', 8080), help='Server port number.')

    args = arg_parser.parse_args()
    #app.run(port=args.port)

    #model.load(args.model)
  
    model.update_path(args.model)
    app.run(debug = True)




if __name__ == '__main__':

    main() 
