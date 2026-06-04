#%%

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({
        "message": "This is the first home page........"
    })

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return jsonify({
            "message": "Predict endpoint working"
        })

    data = request.get_json()

    input_data = pd.DataFrame([{
        "age": data["age"],
        "bmi": data["bmi"],
        "blood_pressure": data["blood_pressure"],
        "glucose_level": data["glucose_level"]
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return jsonify({
        "prediction": "Yes" if prediction == 1 else "No",
        "risk_probability": round(float(probability), 2)
    })

print(app.url_map)

if __name__ == "__main__":
    app.run(debug=True)