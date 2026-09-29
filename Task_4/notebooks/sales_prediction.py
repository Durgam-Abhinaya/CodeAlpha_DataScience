import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\dataset\Advertising.csv"
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
# REMOVE UNNECESSARY COLUMN
# ============================================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\n========== AFTER REMOVING UNNECESSARY COLUMN ==========")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# ADVERTISING STATISTICS
# ============================================================

print("\n========== ADVERTISING STATISTICS ==========")

print("\nAverage TV Advertising:")
print(round(df["TV"].mean(), 2))

print("\nAverage Radio Advertising:")
print(round(df["Radio"].mean(), 2))

print("\nAverage Newspaper Advertising:")
print(round(df["Newspaper"].mean(), 2))

print("\nAverage Sales:")
print(round(df["Sales"].mean(), 2))


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n========== CORRELATION MATRIX ==========")

correlation = df.corr()

print(correlation)


# ============================================================
# SALES DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    df["Sales"],
    bins=15
)

plt.xlabel("Sales")

plt.ylabel("Frequency")

plt.title("Distribution of Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\outputs\sales_distribution.png"
)

plt.close()


# ============================================================
# TV ADVERTISING VS SALES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["TV"],
    df["Sales"]
)

plt.xlabel("TV Advertising Spend")

plt.ylabel("Sales")

plt.title("TV Advertising vs Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\outputs\tv_vs_sales.png"
)

plt.close()


# ============================================================
# RADIO ADVERTISING VS SALES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Radio"],
    df["Sales"]
)

plt.xlabel("Radio Advertising Spend")

plt.ylabel("Sales")

plt.title("Radio Advertising vs Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\outputs\radio_vs_sales.png"
)

plt.close()


# ============================================================
# NEWSPAPER ADVERTISING VS SALES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Newspaper"],
    df["Sales"]
)

plt.xlabel("Newspaper Advertising Spend")

plt.ylabel("Sales")

plt.title("Newspaper Advertising vs Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\outputs\newspaper_vs_sales.png"
)

plt.close()


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\outputs\correlation_matrix.png"
)

plt.close()


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[
    [
        "TV",
        "Radio",
        "Newspaper"
    ]
]

y = df["Sales"]


print("\n========== FEATURES AND TARGET ==========")

print("Features:")
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
# LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# MODEL TRAINING
# ============================================================

print("\n========== MODEL TRAINING ==========")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# MODEL COEFFICIENTS
# ============================================================

print("\n========== MODEL COEFFICIENTS ==========")

print("Intercept:", round(model.intercept_, 4))

print(
    "TV coefficient:",
    round(model.coef_[0], 4)
)

print(
    "Radio coefficient:",
    round(model.coef_[1], 4)
)

print(
    "Newspaper coefficient:",
    round(model.coef_[2], 4)
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\n========== PREDICTIONS ==========")

results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
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

print(
    "Mean Absolute Error (MAE):",
    round(mae, 4)
)

print(
    "Mean Squared Error (MSE):",
    round(mse, 4)
)

print(
    "Root Mean Squared Error (RMSE):",
    round(rmse, 4)
)

print(
    "R² Score:",
    round(r2, 4)
)


# ============================================================
# ACTUAL VS PREDICTED SALES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\outputs\actual_vs_predicted_sales.png"
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

plt.xlabel("Predicted Sales")

plt.ylabel("Residual")

plt.title("Residual Analysis")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\outputs\residual_analysis.png"
)

plt.close()


# ============================================================
# ADVERTISING IMPACT
# ============================================================

print("\n========== ADVERTISING IMPACT ==========")

print(
    "TV coefficient:",
    round(model.coef_[0], 4)
)

print(
    "Radio coefficient:",
    round(model.coef_[1], 4)
)

print(
    "Newspaper coefficient:",
    round(model.coef_[2], 4)
)

print(
    "\nThe coefficients represent the estimated change in sales "
    "for a one-unit increase in the corresponding advertising "
    "spend, while keeping the other advertising variables constant."
)


# ============================================================
# PREDICT NEW SALES
# ============================================================

new_advertising = pd.DataFrame({
    "TV": [150],
    "Radio": [30],
    "Newspaper": [20]
})

predicted_sales = model.predict(
    new_advertising
)

print("\n========== NEW SALES PREDICTION ==========")

print(
    "TV Advertising:",
    new_advertising["TV"].iloc[0]
)

print(
    "Radio Advertising:",
    new_advertising["Radio"].iloc[0]
)

print(
    "Newspaper Advertising:",
    new_advertising["Newspaper"].iloc[0]
)

print(
    "Predicted Sales:",
    round(predicted_sales[0], 2)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print(
    "Total observations:",
    len(df)
)

print(
    "Average Sales:",
    round(df["Sales"].mean(), 2)
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
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_4\outputs"
)