from flask import Flask,render_template,request
import pickle
import sklearn
import numpy as np
import pandas as pd
app = Flask(__name__)

filename = "knn_model.pkl"
with open(filename, 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template(["index.html"])

@app.route("/predict",methods = ["GET","POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    pred = [np.array(float_features)]
    prediction = model.predict(pred)
    
    return render_template("index.html", output='prediction')




if __name__ == "__main__":
    app.run(debug = True)