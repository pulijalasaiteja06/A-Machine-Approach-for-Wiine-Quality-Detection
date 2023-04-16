from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle

import pandas as pd

app = Flask(__name__)
model = pickle.load(open("abcde.pkl","rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        #now get details from web page using POST Method by using respnose
        fixed_Acidity = request.form["fixed_acidity"]
        volatile_Acidity = request.form["volatile_acidity"]
        citric_Acid = request.form["citric_acid"]
        residual_Sugar = request.form["residual_sugar"]
        Chlorides = request.form["chlorides"]
        free_Sulfer_Dioxide = request.form["free_sulfer_dioxide"]
        total_Sulfer_Dioxide=request.form["total_sulfer_dioxide"]
        Density = request.form["density"]
        PH = request.form["pH"]
        Sulphates = request.form["sulphates"]
        Alcohol = request.form["alcohol"]
        
        #Passing values to the Model
        prediction=model.predict([[fixed_Acidity,volatile_Acidity,citric_Acid,residual_Sugar,Chlorides,free_Sulfer_Dioxide,total_Sulfer_Dioxide,Density,PH,Sulphates,Alcohol]])
        if prediction[0]==1:
            result="Good"
        else:
             result="Bad"
        return render_template('home.html',prediction_text="{} quality wine".format(result))

    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
