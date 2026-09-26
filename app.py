from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

import pandas as pd
import joblib
import os


# ==========================================
# APP
# ==========================================

app = Flask(__name__)


# ==========================================
# SETTINGS
# ==========================================

CURRENT_YEAR = 2026

DATASET_FILE = "Dataset.csv"
MODEL_FILE = "model.pkl"


# ==========================================
# LOAD MODEL
# ==========================================

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        "model.pkl not found. "
        "Run: python model.py"
    )


model = joblib.load(
    MODEL_FILE
)


# ==========================================
# LOAD DATASET
# ==========================================

if not os.path.exists(DATASET_FILE):

    raise FileNotFoundError(
        "Dataset.csv not found."
    )


dataset = pd.read_csv(
    DATASET_FILE
)


# ==========================================
# FEATURE ENGINEERING
# ==========================================

dataset["year"] = pd.to_numeric(
    dataset["year"],
    errors="coerce"
)

dataset["distance_travelled(kms)"] = pd.to_numeric(
    dataset["distance_travelled(kms)"],
    errors="coerce"
)

dataset["brand_rank"] = pd.to_numeric(
    dataset["brand_rank"],
    errors="coerce"
)


dataset["car_age"] = (
    CURRENT_YEAR - dataset["year"]
)


def get_usage_level(distance):

    if pd.isna(distance):
        return "Unknown"

    if distance <= 30000:
        return "Low"

    elif distance <= 60000:
        return "Medium"

    elif distance <= 100000:
        return "High"

    else:
        return "Very High"


dataset["usage_level"] = (
    dataset[
        "distance_travelled(kms)"
    ].apply(get_usage_level)
)


# ==========================================
# HELPER
# ==========================================

def unique_values(column):

    if column not in dataset.columns:
        return []

    values = (
        dataset[column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    values.sort(
        key=lambda x: x.lower()
    )

    return values


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return send_from_directory(
        ".",
        "index.html"
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():

    return jsonify({
        "success": True,
        "message": "Server is running"
    })


# ==========================================
# BRANDS
# ==========================================

@app.route("/api/brands")
def brands():

    return jsonify(
        unique_values("brand")
    )


# ==========================================
# MODELS
# ==========================================

@app.route("/api/models")
def models():

    brand = request.args.get(
        "brand",
        ""
    ).strip()


    if not brand:

        return jsonify([])


    data = dataset[
        dataset["brand"]
        .astype(str)
        .str.strip()
        .str.lower()
        == brand.lower()
    ]


    values = (
        data["model_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    values.sort(
        key=lambda x: x.lower()
    )


    return jsonify(values)


# ==========================================
# FULL MODEL
# ==========================================

@app.route("/api/full-models")
def full_models():

    brand = request.args.get(
        "brand",
        ""
    ).strip()

    model_name = request.args.get(
        "model",
        ""
    ).strip()


    if not brand or not model_name:

        return jsonify([])


    data = dataset[
        (
            dataset["brand"]
            .astype(str)
            .str.strip()
            .str.lower()
            == brand.lower()
        )
        &
        (
            dataset["model_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == model_name.lower()
        )
    ]


    values = (
        data["full_model_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    values.sort(
        key=lambda x: x.lower()
    )


    return jsonify(values)


# ==========================================
# FUEL TYPES
# ==========================================

@app.route("/api/fuel-types")
def fuel_types():

    brand = request.args.get(
        "brand",
        ""
    ).strip()

    model_name = request.args.get(
        "model",
        ""
    ).strip()

    full_model = request.args.get(
        "full_model",
        ""
    ).strip()


    data = dataset[
        (
            dataset["brand"]
            .astype(str)
            .str.strip()
            .str.lower()
            == brand.lower()
        )
        &
        (
            dataset["model_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == model_name.lower()
        )
        &
        (
            dataset["full_model_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == full_model.lower()
        )
    ]


    values = (
        data["fuel_type"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    values.sort(
        key=lambda x: x.lower()
    )


    return jsonify(values)


# ==========================================
# CITIES
# ==========================================

@app.route("/api/cities")
def cities():

    brand = request.args.get(
        "brand",
        ""
    ).strip()

    model_name = request.args.get(
        "model",
        ""
    ).strip()

    full_model = request.args.get(
        "full_model",
        ""
    ).strip()

    fuel = request.args.get(
        "fuel",
        ""
    ).strip()


    data = dataset[
        (
            dataset["brand"]
            .astype(str)
            .str.strip()
            .str.lower()
            == brand.lower()
        )
        &
        (
            dataset["model_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == model_name.lower()
        )
        &
        (
            dataset["full_model_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == full_model.lower()
        )
        &
        (
            dataset["fuel_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            == fuel.lower()
        )
    ]


    values = (
        data["city"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    values.sort(
        key=lambda x: x.lower()
    )


    return jsonify(values)


# ==========================================
# PREDICT
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No data received."
            }), 400


        # ----------------------------------
        # INPUT
        # ----------------------------------

        year = int(
            data["year"]
        )

        brand = str(
            data["brand"]
        ).strip()

        model_name = str(
            data["model_name"]
        ).strip()

        full_model_name = str(
            data["full_model_name"]
        ).strip()

        distance = float(
            data["distance_travelled"]
        )

        fuel_type = str(
            data["fuel_type"]
        ).strip()

        city = str(
            data["city"]
        ).strip()


        # ----------------------------------
        # VALIDATION
        # ----------------------------------

        if year < 1990 or year > CURRENT_YEAR:

            return jsonify({
                "success": False,
                "error": "Invalid manufacturing year."
            }), 400


        if distance < 0:

            return jsonify({
                "success": False,
                "error": "Distance cannot be negative."
            }), 400


        if not brand:

            return jsonify({
                "success": False,
                "error": "Brand is required."
            }), 400


        if not model_name:

            return jsonify({
                "success": False,
                "error": "Model is required."
            }), 400


        if not full_model_name:

            return jsonify({
                "success": False,
                "error": "Full model is required."
            }), 400


        if not fuel_type:

            return jsonify({
                "success": False,
                "error": "Fuel type is required."
            }), 400


        if not city:

            return jsonify({
                "success": False,
                "error": "City is required."
            }), 400


        # ----------------------------------
        # VALIDATE COMBINATION
        # ----------------------------------

        matching = dataset[
            (
                dataset["brand"]
                .astype(str)
                .str.strip()
                .str.lower()
                == brand.lower()
            )
            &
            (
                dataset["model_name"]
                .astype(str)
                .str.strip()
                .str.lower()
                == model_name.lower()
            )
            &
            (
                dataset["full_model_name"]
                .astype(str)
                .str.strip()
                .str.lower()
                == full_model_name.lower()
            )
            &
            (
                dataset["fuel_type"]
                .astype(str)
                .str.strip()
                .str.lower()
                == fuel_type.lower()
            )
            &
            (
                dataset["city"]
                .astype(str)
                .str.strip()
                .str.lower()
                == city.lower()
            )
        ]


        if matching.empty:

            return jsonify({
                "success": False,
                "error":
                "Selected car combination is not available in the dataset."
            }), 400


        # ----------------------------------
        # BRAND RANK
        # ----------------------------------

        brand_rank = float(
            matching.iloc[0]["brand_rank"]
        )


        # ----------------------------------
        # CAR AGE
        # ----------------------------------

        car_age = (
            CURRENT_YEAR - year
        )


        # ----------------------------------
        # USAGE LEVEL
        # ----------------------------------

        usage_level = get_usage_level(
            distance
        )


        # ----------------------------------
        # MODEL INPUT
        # ----------------------------------

        input_data = pd.DataFrame([{

            "year":
                year,

            "brand":
                brand,

            "full_model_name":
                full_model_name,

            "model_name":
                model_name,

            "distance_travelled(kms)":
                distance,

            "fuel_type":
                fuel_type,

            "city":
                city,

            "brand_rank":
                brand_rank,

            "car_age":
                car_age,

            "usage_level":
                usage_level

        }])


        # ----------------------------------
        # PREDICTION
        # ----------------------------------

        prediction = model.predict(
            input_data
        )[0]


        prediction = max(
            0,
            float(prediction)
        )


        # ----------------------------------
        # RESPONSE
        # ----------------------------------

        return jsonify({

            "success":
                True,

            "predicted_price":
                round(
                    prediction,
                    2
                ),

            "formatted_price":
                f"₹{prediction:,.0f}"

        })


    except KeyError as error:

        return jsonify({

            "success": False,

            "error":
                f"Missing field: {error}"

        }), 400


    except Exception as error:

        print(
            "Prediction Error:",
            error
        )

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print(" USED CAR PRICE PREDICTION")
    print(" Flask Backend Started")
    print("======================================")
    print(" http://127.0.0.1:5000")
    print("======================================")
    print("")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )