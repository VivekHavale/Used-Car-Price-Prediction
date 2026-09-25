import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

DATASET_PATH = "Dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ==========================================
# 2. Remove Unnecessary Columns
# ==========================================

columns_to_drop = [
    "Unnamed: 0",
    "Id",

    # Existing engineered columns that can cause
    # target leakage or are not needed for prediction
    "inv_car_price",
    "inv_car_dist",
    "inv_car_age",
    "inv_brand",
    "std_invprice",
    "std_invdistance_travelled",
    "std_invrank",
    "best_buy1",
    "best_buy2"
]

df = df.drop(
    columns=[col for col in columns_to_drop if col in df.columns]
)


# ==========================================
# 3. Feature Engineering
# ==========================================

# Calculate car age from manufacturing year
current_year = 2026

df["car_age"] = current_year - df["year"]

# Make sure age cannot be negative
df["car_age"] = df["car_age"].clip(lower=0)


# Create a simple usage category
df["usage_level"] = pd.cut(
    df["distance_travelled(kms)"],
    bins=[-1, 30000, 60000, 100000, np.inf],
    labels=["Low", "Medium", "High", "Very High"]
)


# ==========================================
# 4. Remove Rows With Missing Target
# ==========================================

df = df.dropna(subset=["price"])


# ==========================================
# 5. Define Input and Target
# ==========================================

X = df.drop("price", axis=1)
y = df["price"]


# ==========================================
# 6. Select Numerical and Categorical Columns
# ==========================================

numerical_features = [
    "year",
    "distance_travelled(kms)",
    "brand_rank",
    "car_age"
]

categorical_features = [
    "brand",
    "full_model_name",
    "model_name",
    "fuel_type",
    "city",
    "usage_level"
]


# Keep only columns that actually exist
numerical_features = [
    col for col in numerical_features
    if col in X.columns
]

categorical_features = [
    col for col in categorical_features
    if col in X.columns
]


# ==========================================
# 7. Numerical Pipeline
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ==========================================
# 8. Categorical Pipeline
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ==========================================
# 9. Combine Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================
# 10. Create Linear Regression Model
# ==========================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ==========================================
# 11. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 12. Train Model
# ==========================================

print("\nTraining Linear Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 13. Make Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 14. Evaluate Model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print(f"MAE  : ₹{mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")

print("===================================")


# ==========================================
# 15. Save Complete Pipeline
# ==========================================

MODEL_PATH = "model.pkl"

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully as: {MODEL_PATH}")


# ==========================================
# 16. Test One Prediction
# ==========================================

sample_car = pd.DataFrame([
    {
        "year": 2018,
        "brand": "Honda",
        "full_model_name": "Honda Brio S MT",
        "model_name": "Brio",
        "distance_travelled(kms)": 30000,
        "fuel_type": "Petrol",
        "city": "Mumbai",
        "brand_rank": 7,
        "car_age": 2026 - 2018,
        "usage_level": "Medium"
    }
])


sample_prediction = model.predict(sample_car)[0]

print("\n===================================")
print("SAMPLE PREDICTION")
print("===================================")
print(f"Estimated Price: ₹{sample_prediction:,.2f}")
print("===================================")