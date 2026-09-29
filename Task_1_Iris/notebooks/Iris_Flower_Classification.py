import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("C:\\Users\\durga\\OneDrive\\Desktop\\codeAlpha\\Task_1_Iris\\dataset\\Iris.csv")

print("\n========== DATASET ==========")
print(df.head())


# ============================================================
# 2. UNDERSTAND THE DATASET
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
# 3. DATA CLEANING
# ============================================================

print("\n========== DATA CLEANING ==========")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)


# ============================================================
# 4. SPECIES DISTRIBUTION
# ============================================================

print("\n========== SPECIES DISTRIBUTION ==========")
print(df["Species"].value_counts())


plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Species"
)

plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("outputs/species_distribution.png")
plt.show()


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="SepalWidthCm",
    hue="Species"
)

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")

plt.tight_layout()
plt.savefig("outputs/sepal_scatter.png")
plt.show()


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.tight_layout()
plt.savefig("outputs/petal_scatter.png")
plt.show()


# ============================================================
# 6. PAIR PLOT
# ============================================================

sns.pairplot(
    df,
    hue="Species"
)

plt.savefig("outputs/pairplot.png")
plt.show()


# ============================================================
# 7. BOX PLOT
# ============================================================

features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df[features]
)

plt.title("Distribution of Iris Flower Measurements")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("outputs/boxplot.png")
plt.show()


# ============================================================
# 8. CORRELATION MATRIX
# ============================================================

correlation = df[features].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.show()


# ============================================================
# 9. PREPARE FEATURES AND TARGET
# ============================================================

X = df[features]

y = df["Species"]


print("\n========== FEATURES AND TARGET ==========")

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 10. ENCODE TARGET LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\n========== LABEL ENCODING ==========")
print("Species classes:")
print(label_encoder.classes_)


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 12. CREATE MACHINE LEARNING MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\n========== MODEL TRAINING ==========")

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 15. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

plt.figure(figsize=(8, 5))

feature_importance.plot(
    kind="bar"
)

plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()


# ============================================================
# 19. TEST WITH A NEW FLOWER
# ============================================================

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=features
)

prediction = model.predict(
    new_flower
)

predicted_species = label_encoder.inverse_transform(
    prediction
)

print("\n========== NEW FLOWER PREDICTION ==========")
print("Flower measurements:")
print(new_flower)

print("\nPredicted Species:", predicted_species[0])


# ============================================================
# 20. FINAL RESULT
# ============================================================

print("\n==========================================")
print("          FINAL RESULTS")
print("==========================================")

print("Dataset size:", df.shape)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model: Random Forest Classifier")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("==========================================")