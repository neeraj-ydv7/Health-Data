from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("ml/diabetes_model.joblib")

# 🔹 Add mapping here
mapping = {
    "Pregnancies": "preg",
    "Glucose": "plas",
    "BloodPressure": "pres",
    "SkinThickness": "skin",
    "Insulin": "insu",
    "BMI": "mass",
    "DiabetesPedigreeFunction": "pedi",
    "Age": "age"
}

@app.route("/")
def home():
    return "Flask API is running!"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# 🔹 Replace your old /predict function with this one
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Map incoming keys to training keys
    mapped_data = {mapping[k]: v for k, v in data.items()}
    df = pd.DataFrame([mapped_data])

    prediction = model.predict(df)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
