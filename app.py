from flask import Flask, request, jsonify
from flask import render_template

import pickle

app = Flask(__name__)

# Load trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "House Price Prediction API is running 🚀"

@app.route("/predict-test", methods=["GET"])
def predict_test():
    return "predict-test route works"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    def yes_no(val):
        return 1 if val.lower() == "yes" else 0

    furnishing_semi = 1 if data["furnishingstatus"] == "semi-furnished" else 0
    furnishing_unfurnished = 1 if data["furnishingstatus"] == "unfurnished" else 0

    features = [[
        data["area"],
        data["bedrooms"],
        data["bathrooms"],
        data["stories"],
        yes_no(data["mainroad"]),
        yes_no(data["guestroom"]),
        yes_no(data["basement"]),
        yes_no(data["hotwaterheating"]),
        yes_no(data["airconditioning"]),
        data["parking"],
        yes_no(data["prefarea"]),
        furnishing_semi,
        furnishing_unfurnished
    ]]

    prediction = model.predict(features)

    return jsonify({
        "predicted_price": round(float(prediction[0]), 2)
    })

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/predict-form", methods=["POST"])
def predict_form():
    data = request.form

    def yes_no(val):
        return 1 if val == "yes" else 0

    furnishing_semi = 1 if data["furnishingstatus"] == "semi-furnished" else 0
    furnishing_unfurnished = 1 if data["furnishingstatus"] == "unfurnished" else 0

    features = [[
        int(data["area"]),
        int(data["bedrooms"]),
        int(data["bathrooms"]),
        int(data["stories"]),
        yes_no(data["mainroad"]),
        yes_no(data["guestroom"]),
        yes_no(data["basement"]),
        yes_no(data["hotwaterheating"]),
        yes_no(data["airconditioning"]),
        int(data["parking"]),
        yes_no(data["prefarea"]),
        furnishing_semi,
        furnishing_unfurnished
    ]]

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction=round(float(prediction), 2)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

