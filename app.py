import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("classifier_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text=f"{prediction[0]}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007, debug=True)