import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_3\dataset\car data.csv"
)

print("\n========== DATASET ==========")
print(df.head())


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# DATA CLEANING
# ============================================================

print("\n========== DATA CLEANING ==========")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n========== CATEGORICAL FEATURES ==========")

print("\nFuel Type:")
print(df["Fuel_Type"].value_counts())

print("\nSelling Type:")
print(df["Selling_type"].value_counts())

print("\nTransmission:")
print(df["Transmission"].value_counts())

print("\nOwner:")
print(df["Owner"].value_counts())


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["Car_Age"] = 2026 - df["Year"]

print("\n========== FEATURE ENGINEERING ==========")

print(df[["Year", "Car_Age"]].head())


# ============================================================
# TARGET AND FEATURES
# ============================================================

X = df[
    [
        "Present_Price",
        "Driven_kms",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
        "Car_Age"
    ]
]

y = df["Selling_Price"]


print("\n========== FEATURES ==========")

print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# PREPROCESSING
# ============================================================

categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numeric_features = [
    "Present_Price",
    "Driven_kms",
    "Owner",
    "Car_Age"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# MODEL TRAINING
# ============================================================

print("\n========== MODEL TRAINING ==========")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)


print("\n========== PREDICTIONS ==========")

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(results.head(10))


# ============================================================
# MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== MODEL EVALUATION ==========")

print("Mean Absolute Error (MAE):", round(mae, 4))

print("Mean Squared Error (MSE):", round(mse, 4))

print("Root Mean Squared Error (RMSE):", round(rmse, 4))

print("R² Score:", round(r2, 4))


# ============================================================
# ACTUAL VS PREDICTED PRICES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Selling Price")

plt.ylabel("Predicted Selling Price")

plt.title("Actual vs Predicted Car Prices")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_3\outputs\actual_vs_predicted.png"
)

plt.close()


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Selling Price")

plt.ylabel("Residual")

plt.title("Residual Analysis")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_3\outputs\residual_analysis.png"
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = pipeline.named_steps[
    "model"
].feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")

print(feature_importance)


plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title("Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_3\outputs\feature_importance.png"
)

plt.close()


# ============================================================
# SELLING PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    df["Selling_Price"],
    bins=20
)

plt.xlabel("Selling Price")

plt.ylabel("Number of Cars")

plt.title("Distribution of Car Selling Prices")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_3\outputs\selling_price_distribution.png"
)

plt.close()


# ============================================================
# PRESENT PRICE VS SELLING PRICE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Present_Price"],
    df["Selling_Price"]
)

plt.xlabel("Present Price")

plt.ylabel("Selling Price")

plt.title("Present Price vs Selling Price")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_3\outputs\present_vs_selling_price.png"
)

plt.close()


# ============================================================
# CAR AGE VS SELLING PRICE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Car_Age"],
    df["Selling_Price"]
)

plt.xlabel("Car Age")

plt.ylabel("Selling Price")

plt.title("Car Age vs Selling Price")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_3\outputs\car_age_vs_selling_price.png"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("Total cars:", len(df))

print(
    "Average selling price:",
    round(df["Selling_Price"].mean(), 2)
)

print(
    "Minimum selling price:",
    round(df["Selling_Price"].min(), 2)
)

print(
    "Maximum selling price:",
    round(df["Selling_Price"].max(), 2)
)

print(
    "MAE:",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R² Score:",
    round(r2, 4)
)

print("\n========== ANALYSIS COMPLETED ==========")

print(
    "All graphs have been saved in:"
)

print(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_3\outputs"
)