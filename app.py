

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
print("APP.PY STARTED")
app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# ==========================================
# Prediction API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Get values from frontend
        year = int(data["year"])
        brand = data["brand"]
        full_model_name = data["full_model_name"]
        model_name = data["model_name"]
        distance = float(data["distance_travelled"])
        fuel_type = data["fuel_type"]
        city = data["city"]
        brand_rank = float(data["brand_rank"])

        # Feature engineering
        car_age = 2026 - year

        if distance <= 30000:
            usage_level = "Low"

        elif distance <= 60000:
            usage_level = "Medium"

        elif distance <= 100000:
            usage_level = "High"

        else:
            usage_level = "Very High"

        # Create DataFrame
        input_data = pd.DataFrame([
            {
                "year": year,
                "brand": brand,
                "full_model_name": full_model_name,
                "model_name": model_name,
                "distance_travelled(kms)": distance,
                "fuel_type": fuel_type,
                "city": city,
                "brand_rank": brand_rank,
                "car_age": car_age,
                "usage_level": usage_level
            }
        ])

        # Predict
        prediction = model.predict(input_data)[0]

        # Prevent negative price
        prediction = max(0, prediction)

        return jsonify({
            "success": True,
            "predicted_price": round(float(prediction), 2),
            "formatted_price": f"₹{prediction:,.0f}"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ==========================================
# Run Flask Server
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )